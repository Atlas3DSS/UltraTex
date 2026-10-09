"""Render the six canonical G-buffer views UltraTex expects from a mesh (Blender script).

Replicates the G-buffer TexVerse canonical rig (calibrated from the dataset's
position maps and poses): a perspective camera with fx = 1.5625 * width at
distance 1.67, six views [front, left, back, right, top, bottom]; the object is
centred and scaled so its longest side is 0.92 (world units in the dataset's
[-1, 1] box). Outputs, per view ``000..005``:

  <out>/bump_normal_world/*.webp  RGBA, world normals stored as (X, Y, -Z) in Blender
                                   coordinates, rgb = (n + 1) / 2  (dataset convention)
  <out>/position/*.webp            RGBA, (world + 1) / 2 in the dataset's y-up frame
  <out>/albedo/*.webp              RGBA placeholder (base colour if the mesh has one)
  <out>/uv/*.png                   16-bit UV coordinates (R, G) + alpha, for back-projection
  <out>/meta.json                  rig parameters

Usage:
  blender -b -P tools/render_gbuffer.py -- --mesh asset.glb --out /path/asset_dir [--res 2048]
"""

import argparse
import json
import math
import os
import sys

import bpy
import mathutils

RIG = [  # (camera position direction, up) in Blender z-up coordinates
    ((0, -1, 0), (0, 0, 1)),  # 000 front  (dataset +z_p)
    ((-1, 0, 0), (0, 0, 1)),  # 001 left   (dataset -x_p)
    ((0, 1, 0), (0, 0, 1)),   # 002 back
    ((1, 0, 0), (0, 0, 1)),   # 003 right
    ((0, 0, 1), (0, 1, 0)),   # 004 top    (image down = dataset +z_p = Blender -Y)
    ((0, 0, -1), (0, 1, 0)),  # 005 bottom
]
DISTANCE = 1.67
FX_OVER_WIDTH = 6400.0 / 4096.0
LONGEST_SIDE = 0.92


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh format {ext}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh objects imported")
    return meshes


def normalize(meshes):
    """Centre the union bounding box at the origin and scale the longest side to LONGEST_SIDE."""
    bpy.context.view_layer.update()
    lo = mathutils.Vector((math.inf,) * 3)
    hi = mathutils.Vector((-math.inf,) * 3)
    for o in meshes:
        for corner in o.bound_box:
            p = o.matrix_world @ mathutils.Vector(corner)
            lo = mathutils.Vector(map(min, lo, p))
            hi = mathutils.Vector(map(max, hi, p))
    centre = (lo + hi) / 2
    size = max(hi - lo)
    scale = LONGEST_SIDE / size
    root = bpy.data.objects.new("ultratex_root", None)
    bpy.context.scene.collection.objects.link(root)
    for o in meshes:
        if o.parent is None or o.parent.type != "MESH":
            top = o
            while top.parent is not None:
                top = top.parent
            if top.parent is None and top is not root:
                top.parent = root
    root.location = -centre * scale
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return {"centre": list(centre), "source_longest_side": size, "scale": scale}


def make_camera(res):
    cam_data = bpy.data.cameras.new("ultratex_cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = FX_OVER_WIDTH * cam_data.sensor_width  # fx = f/sensor * width
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("ultratex_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def aim_camera(cam, pos_dir, up):
    pos = mathutils.Vector(pos_dir) * DISTANCE
    look = -mathutils.Vector(pos_dir)
    up = mathutils.Vector(up)
    right = look.cross(up)
    # Blender camera looks down its local -Z with +Y up
    rot = mathutils.Matrix((right, up, -look)).transposed()
    cam.matrix_world = mathutils.Matrix.Translation(pos) @ rot.to_4x4()


def gbuffer_material(mode):
    """Emission-only material: 'normal' (world normal), 'position' (world position), 'uv' (uv coords)."""
    mat = bpy.data.materials.new(f"gbuffer_{mode}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    if mode == "uv":
        tex = nodes.new("ShaderNodeTexCoord")
        links.new(tex.outputs["UV"], emit.inputs["Color"])
    else:
        geo = nodes.new("ShaderNodeNewGeometry")
        src = geo.outputs["Normal"] if mode == "normal" else geo.outputs["Position"]
        # dataset convention: normal stored = ((X, Y, -Z) + 1) / 2 ; position stored = ((x_p, y_p, z_p) + 1) / 2
        # with (x_p, y_p, z_p) = (X, Z, -Y)
        sep = nodes.new("ShaderNodeSeparateXYZ")
        links.new(src, sep.inputs["Vector"])
        comb = nodes.new("ShaderNodeCombineXYZ")
        if mode == "normal":
            links.new(sep.outputs["X"], comb.inputs["X"])
            links.new(sep.outputs["Y"], comb.inputs["Y"])
            neg = nodes.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1.0
            links.new(sep.outputs["Z"], neg.inputs[0]); links.new(neg.outputs[0], comb.inputs["Z"])
        else:
            links.new(sep.outputs["X"], comb.inputs["X"])
            links.new(sep.outputs["Z"], comb.inputs["Y"])
            neg = nodes.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1.0
            links.new(sep.outputs["Y"], neg.inputs[0]); links.new(neg.outputs[0], comb.inputs["Z"])
        add = nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"; add.inputs[1].default_value = (1, 1, 1)
        scale = nodes.new("ShaderNodeVectorMath"); scale.operation = "SCALE"; scale.inputs["Scale"].default_value = 0.5
        links.new(comb.outputs[0], add.inputs[0]); links.new(add.outputs[0], scale.inputs[0])
        links.new(scale.outputs[0], emit.inputs["Color"])
    links.new(emit.outputs[0], out.inputs["Surface"])
    return mat


def albedo_material_from(original):
    """Keep the mesh's own base colour (texture or constant) as flat emission, for the placeholder albedo."""
    mat = bpy.data.materials.new("gbuffer_albedo")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    colour = (0.5, 0.5, 0.5, 1.0)
    tex_image = None
    if original is not None and original.use_nodes:
        for n in original.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                base = n.inputs["Base Color"]
                colour = tuple(base.default_value)
                if base.is_linked and base.links[0].from_node.type == "TEX_IMAGE":
                    tex_image = base.links[0].from_node.image
    if tex_image is not None:
        tex = nodes.new("ShaderNodeTexImage"); tex.image = tex_image
        links.new(tex.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = colour
    links.new(emit.outputs[0], out.inputs["Surface"])
    return mat


def assign(meshes, mat):
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)


def setup_render(res, engine):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.filter_size = 0.0 if engine == "CYCLES" else 1.5
    # G-buffers are data: bypass colour management entirely (Standard would apply the sRGB OETF)
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.render.dither_intensity = 0.0
    if engine == "CYCLES":
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 0
        scene.cycles.device = "CPU"
    world = bpy.data.worlds.new("w"); scene.world = world


def render_to(path, fmt, color_depth="8", exr_float=False):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = fmt
    scene.render.image_settings.color_mode = "RGBA"
    if fmt == "OPEN_EXR":
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
    else:
        scene.render.image_settings.color_depth = color_depth
        if fmt == "PNG":
            scene.render.image_settings.compression = 15
        if fmt == "WEBP":
            scene.render.image_settings.quality = 100
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    clear_scene()
    meshes = import_mesh(args.mesh)
    original_mat = next((m for o in meshes for m in o.data.materials if m is not None), None)
    norm = normalize(meshes)
    setup_render(args.res, args.engine)
    cam = make_camera(args.res)
    os.makedirs(args.out, exist_ok=True)
    passes = [  # webp: the UltraTex dataset loaders open 00k.webp
        ("bump_normal_world", gbuffer_material("normal"), "WEBP"),
        ("position", gbuffer_material("position"), "WEBP"),
        ("albedo", albedo_material_from(original_mat), "WEBP"),
        ("uv", gbuffer_material("uv"), "PNG16"),
    ]
    for name, mat, fmt in passes:
        assign(meshes, mat)
        for i, (pos, up) in enumerate(RIG):
            aim_camera(cam, pos, up)
            ext = {"OPEN_EXR": "exr", "WEBP": "webp"}.get(fmt, "png")
            if fmt == "PNG16":
                render_to(os.path.join(args.out, name, f"{i:03d}.png"), "PNG", color_depth="16")
            else:
                render_to(os.path.join(args.out, name, f"{i:03d}.{ext}"), fmt)
    with open(os.path.join(args.out, "meta.json"), "wt") as f:
        json.dump({"mesh": os.path.abspath(args.mesh), "res": args.res, "distance": DISTANCE, "fx_over_width": FX_OVER_WIDTH,
                   "longest_side": LONGEST_SIDE, "normalize": norm, "rig": RIG}, f, indent=2)
    print("GBUFFER_OK", args.out)


if __name__ == "__main__":
    main()

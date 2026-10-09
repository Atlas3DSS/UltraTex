"""Blender script: load a mesh, replace its base colour with a baked texture, export GLB.

  blender -b -P tools/apply_texture.py -- --mesh in.glb --texture texture.png --out out.glb
"""

import argparse
import os
import sys

import bpy


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main():
    args = parse()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(args.mesh)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=args.mesh)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=args.mesh)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=args.mesh)
    else:
        raise SystemExit(f"unsupported {ext}")
    image = bpy.data.images.load(os.path.abspath(args.texture))
    image.pack()
    mat = bpy.data.materials.new("ultratex_albedo")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(mat)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(args.out), export_format="GLB", export_image_format="AUTO")
    print("APPLY_OK", args.out)


if __name__ == "__main__":
    main()

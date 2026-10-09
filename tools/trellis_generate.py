#!/usr/bin/env python
"""Generate a mesh from a reference image through the local TRELLIS Studio API.

  python tools/trellis_generate.py --image concept.png --name fox --out /dir [--resolution 1024] [--gpu John]

Uploads the image, queues one `generate` job, polls /api/state until it
completes, downloads the asset GLB to <out>/<name>.glb and prints its path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

API = os.environ.get("TRELLIS_API", "http://127.0.0.1:8785")


def multipart(path: str):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def call(path: str, data: bytes | None = None, content_type: str | None = None, method: str | None = None):
    req = urllib.request.Request(API + path, data=data, method=method or ("POST" if data is not None else "GET"))
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation", type=int, default=250000)
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    body, ctype = multipart(args.image)
    upload = call("/api/uploads", body, ctype)
    print("uploaded", upload.get("id"), upload.get("kind"))
    req = {"kind": "generate", "image_id": upload["id"], "resolution": args.resolution, "seed": args.seed, "steps": args.steps,
           "texture_size": args.texture_size, "decimation_target": args.decimation, "gpu": args.gpu, "name": args.name, "variants": 1}
    jobs = call("/api/jobs", json.dumps(req).encode(), "application/json")["jobs"]
    job_id = jobs[0]["id"]
    print("job", job_id, jobs[0].get("status"))
    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        state = call("/api/state")
        job = next((j for j in state["jobs"] if j["id"] == job_id), None)
        if job is None:
            raise SystemExit("job vanished")
        stamp = (job.get("status"), job.get("stage"), job.get("progress"))
        if stamp != last:
            print("  ", *stamp, flush=True)
            last = stamp
        if job["status"] == "completed":
            asset = next((a for a in state["assets"] if a.get("job_id") == job_id), None)
            if asset is None:
                raise SystemExit("completed but no asset recorded")
            os.makedirs(args.out, exist_ok=True)
            dst = os.path.join(args.out, f"{args.name}.glb")
            urllib.request.urlretrieve(API + asset["url"], dst)
            print("TRELLIS_OK", dst)
            return
        if job["status"] in ("failed", "cancelled"):
            raise SystemExit(f"job {job['status']}: {job.get('error')}")
        time.sleep(10)
    raise SystemExit("timed out")


if __name__ == "__main__":
    main()

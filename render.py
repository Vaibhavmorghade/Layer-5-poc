"""render.py (Person 3): renders camera frames through viewer/index.html.

Usage (run from the layerfive folder):
    python render.py out/orbit/cams.json --limit 3
    python render.py out/orbit/cams.json
    python render.py out/orbit/cams.json --size 960x600     # if too slow

Person 4 reuses render_frames(..., mode="id") for the ID masks.
"""
import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER_DIR = os.path.join(HERE, "viewer")

CHROMIUM_ARGS = [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]


def load_cams(path):
    with open(path) as f:
        return json.load(f)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(port):
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=VIEWER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/index.html"
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=1)
            return proc
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("viewer server did not start")


def render_frames(cams, out_dir, mode="rgb", subdir=None):
    """Render each camera and save PNGs to out_dir/<subdir>/{frame_id:06d}.png.

    mode   : "rgb" or "id" (passed to the viewer as ?mode=)
    subdir : defaults to mode ("rgb" or "id")
    """
    subdir = subdir or mode
    target = os.path.join(out_dir, subdir)
    os.makedirs(target, exist_ok=True)

    port = _free_port()
    server = _start_server(port)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=CHROMIUM_ARGS)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("console", lambda m: print("  [browser]", m.type, m.text))
            page.on("pageerror", lambda e: print("  [page error]", e))
            page.goto(f"http://127.0.0.1:{port}/index.html?mode={mode}",wait_until="domcontentloaded", timeout=120000)
            page.wait_for_function(
                "window.sceneReady === true || window.sceneError !== null",
                timeout=180000,
            )
            err = page.evaluate("window.sceneError")
            if err:
                raise RuntimeError(f"viewer failed to load the scene: {err}")

            t0 = time.time()
            for n, cam in enumerate(cams):
                page.set_viewport_size({"width": cam["width"], "height": cam["height"]})
                data_url = page.evaluate("c => window.renderFrame(c)", cam)
                png = base64.b64decode(data_url.split(",", 1)[1])
                path = os.path.join(target, f"{cam['frame_id']:06d}.png")
                with open(path, "wb") as f:
                    f.write(png)
                print(f"  saved {path}  ({n + 1}/{len(cams)}, "
                      f"{time.time() - t0:.1f}s elapsed)")
            browser.close()
    finally:
        server.terminate()
    return target


def render_rgb(cams, out_dir):
    """Contract function: saves out_dir/rgb/{frame_id:06d}.png."""
    return render_frames(cams, out_dir, mode="rgb", subdir="rgb")


def make_video(out_dir, fps=10):
    """ffmpeg -framerate 10 -i rgb/%06d.png -pix_fmt yuv420p drone_path.mp4"""
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(out_dir, "rgb", "%06d.png"),
        "-pix_fmt", "yuv420p",
        os.path.join(out_dir, "drone_path.mp4"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cams_json")
    ap.add_argument("--mode", default="rgb", choices=["rgb", "id"])
    ap.add_argument("--limit", type=int, default=None, help="render only the first N frames")
    ap.add_argument("--size", default=None, help="override size, e.g. 960x600")
    ap.add_argument("--video", action="store_true", help="make drone_path.mp4 afterwards")
    args = ap.parse_args()

    cams = load_cams(args.cams_json)
    if args.limit:
        cams = cams[: args.limit]
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
        for c in cams:
            c["width"], c["height"] = w, h

    out_dir = os.path.dirname(os.path.abspath(args.cams_json))
    render_frames(cams, out_dir, mode=args.mode)
    if args.video and args.mode == "rgb":
        make_video(out_dir)

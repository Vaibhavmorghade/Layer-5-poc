"""Person 2: camera trajectories and 3D -> 2D projection.

Contract: Y-up, metres, glTF world. Ground is at y ~= 97, scene centre at
x ~= 115, z ~= -76. Camera frame i shows manifest frame i % num_frames
(manifest is 10 fps, 120-frame loop).

Camera dict:
    {frame_id, t, pos:[x,y,z], look_at:[x,y,z], fov_deg, width, height,
     K (3x3 list), w2c (4x4 list)}
    fov_deg is the VERTICAL field of view.
    w2c is world -> camera in OpenCV convention (x right, y down, z forward).
    t is the time in seconds inside the 12 s loop: (frame_id % 120) / 10.
    That is what Person 3 passes to mixer.setTime(t).

Public API:
    look_at(pos, target)                  -> 4x4 w2c
    intrinsics(width, height, fov_deg)    -> 3x3 K
    make_camera(frame_id, pos, target)    -> camera dict
    project(points, cam)                  -> (uv Nx2, depth N)
    make_trajectory(kind, n_frames=60, m=None) -> list of camera dicts
    save_cams(kind, cams, out_root="out") -> writes out/<kind>/cams.json
"""
import json
import os

import numpy as np

WIDTH, HEIGHT, FOV_DEG = 1280, 800, 60.0
GROUND_Y = 97.0
CENTER = np.array([115.0, GROUND_Y, -76.0])
FPS, NUM_FRAMES = 10, 120
KINDS = ("orbit", "flyover", "lawnmower", "follow")


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------
def intrinsics(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG):
    """K with fx = fy = (height/2) / tan(fov_deg/2), principal point at centre."""
    f = (height / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def look_at(pos, target, up=(0.0, 1.0, 0.0)):
    """4x4 world-to-camera matrix, OpenCV convention (x right, y down, z fwd).

    Rows of the rotation are [right, down, forward]. If the view direction is
    (almost) parallel to `up` (straight-down camera), the image 'up' is set to
    world -Z so the image does not spin.
    """
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(target, dtype=float) - pos
    f = f / np.linalg.norm(f)
    up = np.asarray(up, dtype=float)
    if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:
        up = np.array([0.0, 0.0, -1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image-down
    R = np.stack([r, d, f])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ pos
    return w2c


def make_camera(frame_id, pos, target, width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG):
    """Build one camera dict from the contract."""
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "frame_id": int(frame_id),
        "t": float((frame_id % NUM_FRAMES) / FPS),
        "pos": [float(x) for x in pos],
        "look_at": [float(x) for x in target],
        "fov_deg": float(fov_deg),
        "width": int(width),
        "height": int(height),
        "K": intrinsics(width, height, fov_deg).tolist(),
        "w2c": look_at(pos, target).tolist(),
    }


def project(points, cam):
    """Project Nx3 world points with camera dict `cam`.

    Returns (uv, depth):
        uv    Nx2 pixel coords (u right, v down). Rows for points behind the
              camera are NaN.
        depth N camera-space z. depth <= 0 means behind the camera
              (use `depth > 0` as the valid mask).
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    K = np.asarray(cam["K"], dtype=float)
    w2c = np.asarray(cam["w2c"], dtype=float)
    Xc = P @ w2c[:3, :3].T + w2c[:3, 3]
    depth = Xc[:, 2]
    uv = np.full((len(P), 2), np.nan)
    ok = depth > 1e-6
    uv[ok, 0] = K[0, 0] * Xc[ok, 0] / depth[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * Xc[ok, 1] / depth[ok] + K[1, 2]
    return uv, depth


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------
def _resample_polyline(pts, n):
    """n points at equal arc-length spacing along a polyline (endpoints kept)."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, pts[:, k]) for k in range(3)], axis=1)


def _orbit(n, radius=60.0, height=50.0):
    y = GROUND_Y + height
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = [CENTER[0] + radius * np.cos(a), y, CENTER[2] + radius * np.sin(a)]
        out.append((pos, CENTER))
    return out


def _flyover(n, height=50.0, pitch_deg=60.0, x0=10.0, x1=130.0):
    """Straight line along +x at y = 97 + 50, camera pitched 60 deg down."""
    y = GROUND_Y + height
    p = np.radians(pitch_deg)
    d = np.array([np.cos(p), -np.sin(p), 0.0])
    out = []
    for x in np.linspace(x0, x1, n):
        pos = np.array([x, y, CENTER[2]])
        out.append((pos, pos + 10.0 * d))
    return out


def _lawnmower(n, height=70.0, x_range=(75.0, 155.0), lanes_z=(-110.0, -60.0, -10.0)):
    """Boustrophedon grid at y = 97 + 70, looking straight down.

    Lanes run along x; the footprint at this height is ~129 m (x) by ~81 m (z),
    so three lanes 50 m apart cover the whole scene (z -146..26) with overlap.
    """
    y = GROUND_Y + height
    waypoints = []
    for k, z in enumerate(lanes_z):
        xs = x_range if k % 2 == 0 else x_range[::-1]
        waypoints += [[xs[0], y, z], [xs[1], y, z]]
    path = _resample_polyline(waypoints, n)
    return [(p, p - np.array([0.0, height, 0.0])) for p in path]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _follow(n, m, car_id=20, behind=25.0, above=15.0, max_turn_deg=4.0):
    """25 m behind and 15 m above the car, always looking at its centre.

    "Behind" uses the car's heading (yaw: forward = (cos yaw, 0, -sin yaw)).
    The heading used for the camera turns at most `max_turn_deg` per frame so
    the camera does not teleport when the car reverses direction.
    """
    from loader import boxes_at  # Person 1

    def car_box(i):
        return next(b for b in boxes_at(m, i) if b["object_id"] == car_id)

    out = []
    psi = None  # smoothed heading angle (same convention as yaw)
    for i in range(n):
        b = car_box(i)
        c = np.asarray(b["center"], dtype=float)
        if psi is None:
            psi = b["yaw"]
        else:
            step = np.radians(max_turn_deg)
            psi += np.clip(_wrap(b["yaw"] - psi), -step, step)
        fwd = np.array([np.cos(psi), 0.0, -np.sin(psi)])
        pos = c - behind * fwd + np.array([0.0, above, 0.0])
        out.append((pos, c))
    return out


def make_trajectory(kind, n_frames=60, m=None):
    """List of camera dicts for `kind` in orbit | flyover | lawnmower | follow.

    All use 1280 x 800, fov_deg = 60. `m` (manifest) is only needed for
    'follow'; it is loaded from scene_manifest.json if omitted.
    """
    if kind == "orbit":
        poses = _orbit(n_frames)
    elif kind == "flyover":
        poses = _flyover(n_frames)
    elif kind == "lawnmower":
        poses = _lawnmower(n_frames)
    elif kind == "follow":
        if m is None:
            from loader import load_manifest
            m = load_manifest("scene_manifest.json")
        poses = _follow(n_frames, m)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return [make_camera(i, p, t) for i, (p, t) in enumerate(poses)]


def save_cams(kind, cams, out_root="out"):
    """Dump cameras to out/<kind>/cams.json (Person 3 reads this)."""
    d = os.path.join(out_root, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "cams.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cams, f)
    return path


# --------------------------------------------------------------------------
# Self-test + dump:  python cameras.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from loader import load_manifest, boxes_at, box_corners

    m = load_manifest("scene_manifest.json")

    # intrinsics
    K = intrinsics()
    assert abs(K[0, 0] - 400 / np.tan(np.radians(30))) < 1e-9 and K[0, 2] == 640 and K[1, 2] == 400

    # w2c is a proper rotation + translation, and pos maps to the origin
    cam = make_camera(0, [0, 10, 0], [0, 0, 20])
    w2c = np.array(cam["w2c"])
    R = w2c[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9
    assert np.allclose(w2c @ np.array([0, 10, 0, 1]), [0, 0, 0, 1])

    # target projects to image centre; point behind camera is flagged
    uv, d = project([[0, 0, 20], [0, 10, -5]], cam)
    assert np.allclose(uv[0], [640, 400]) and d[0] > 0
    assert d[1] <= 0 and np.isnan(uv[1]).all()

    # orientation: a point to the world right/up of the view appears right/up
    cam = make_camera(0, [0, 0, 0], [0, 0, -1])  # looking down -Z
    uv, _ = project([[1, 0, -10], [0, 1, -10]], cam)
    assert uv[0, 0] > 640 and uv[1, 1] < 400

    # the required test: follow camera, car 20, frame 0 -> image centre
    follow = make_trajectory("follow", 60, m)
    car = next(b for b in boxes_at(m, 0) if b["object_id"] == 20)
    uv, d = project([car["center"]], follow[0])
    assert np.linalg.norm(uv[0] - [640, 400]) < 2, uv
    print("follow frame 0, car 20 centre ->", np.round(uv[0], 1), "depth", round(d[0], 1))

    # every trajectory: size, fov, frame ids, scene visible, smoothness
    centre_pts = np.array([CENTER + [0, 3, 0]])
    for kind in KINDS:
        cams = make_trajectory(kind, 60, m)
        assert len(cams) == 60 and [c["frame_id"] for c in cams] == list(range(60))
        assert all(c["width"] == 1280 and c["height"] == 800 and c["fov_deg"] == 60 for c in cams)
        pos = np.array([c["pos"] for c in cams])
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        # fraction of frames in which at least one dynamic box centre is in view
        vis = 0
        for c in cams:
            dyn = [b["center"] for b in boxes_at(m, c["frame_id"]) if b["layer"] == "dynamic"]
            uv, dd = project(dyn, c)
            ok = (dd > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 800)
            vis += ok.any()
        print(f"{kind:10s} max step {step.max():5.1f} m  min y {pos[:, 1].min():6.1f}  "
              f"frames with a dynamic object in view: {vis}/60")
        save_cams(kind, cams)
    print("cameras.py OK -> out/<kind>/cams.json for", ", ".join(KINDS))

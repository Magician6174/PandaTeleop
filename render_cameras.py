"""Offscreen camera rendering for the Panda scene.

MuJoCo cameras are declared in the XML:
  - scene.xml worldbody : `front`, `diag`  (fixed third-person views)
  - panda.xml hand body : `wrist`             (eye-in-hand, moves with the arm)

A camera looks down its own -Z axis; `xyaxes="cx cy cz  ux uy uz"` gives the
camera's X (right) and Y (up) axes in the parent frame (-Z = view direction).

Usage:
  python render_cameras.py            # dump RGB + depth PNGs for every camera
  from render_cameras import grab, grab_depth
    grab(cam, model, data)            # -> HxWx3 uint8 RGB
    grab_depth(cam, model, data)      # -> HxW   float32 depth (metres)
"""
import os
import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen GL (egl on Linux/EC2)
HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")


def grab(camera, model, data, renderer=None, h=480, w=640):
    """Render one RGB frame (HxWx3 uint8) from a named camera at the current state."""
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    if own:
        renderer.close()
    return img


def grab_depth(camera, model, data, renderer=None, h=480, w=640):
    """Render a metric depth frame (HxW float32, metres) from a named camera."""
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()          # metres; far plane ~= model extent
    renderer.disable_depth_rendering()
    if own:
        renderer.close()
    return depth


if __name__ == "__main__":
    from PIL import Image
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)

    cams = [model.camera(i).name for i in range(model.ncam)]
    r = mujoco.Renderer(model, height=480, width=640)
    for cam in cams:
        Image.fromarray(grab(cam, model, data, r)).save(os.path.join(HERE, f"cam_{cam}.png"))
        print(f"rendered {cam} -> cam_{cam}.png")
    # also dump a colorized depth map per camera (near = bright)
    for cam in cams:
        d = grab_depth(cam, model, data, r)
        z = np.clip(d, 0.1, 1.5)
        vis = ((1.0 - (z - 0.1) / 1.4) * 255).astype(np.uint8)
        Image.fromarray(vis).save(os.path.join(HERE, f"depth_{cam}.png"))
        print(f"rendered {cam} depth -> depth_{cam}.png "
              f"(range {d[d<d.max()].min():.2f}-{d.max():.2f} m)")
    r.close()

"""Control the Franka Emika Panda (7-DOF + gripper) in MuJoCo.

KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
  ~/miniconda3/envs/python_robotics/bin/python control.py cartesian --record

Modes:
  joints    - direct per-joint HOLD-to-move control (default)
  cartesian - move the gripper in Cartesian (XYZ + roll/pitch/yaw) space via IK
  scripted  - smooth joint-space demo sequence

Run (plain python - uses GLFW directly, no mjpython needed):
  python control.py            # per-joint teleop (default)
  python control.py cartesian  # cartesian-space teleop (IK)
  python control.py scripted   # demo

Per-joint teleop controls:
  1-7        select arm joint
  TAB        cycle selected joint
  UP / DOWN  hold to move selected joint
  ] / [      open / close gripper (hold)
  G          toggle gripper fully closed / open (grasp)
  C          switch the picture-in-picture camera + RGB/depth mode
  P          toggle the picture-in-picture on/off
  F          cycle coordinate-frame axes (world -> sites -> bodies -> off)
  V          toggle the live grip-force gauge
  R          reset to home pose
  SPACE      pause / resume physics
  mouse      orbit | right-drag pan | scroll zoom
  ESC        quit

The Panda's 7 arm actuators are position servos (ctrl = target joint angle);
actuator 8 drives the gripper tendon over ctrl range 0-255 (255 = open).
"""
import os
import sys
import json
import time
import numpy as np
import mujoco
import glfw
import OpenGL.GL as GL

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")
MOVE_RATE = 1.0      # rad/s a held joint key slews the target (per-joint mode)
GRIP_RATE = 200.0    # ctrl units/s for the gripper (range 0-255)
SIM_DT = 0.001       # physics timestep (s); 0.002 was too coarse for the grip

model = mujoco.MjModel.from_xml_path(SCENE)
model.opt.timestep = SIM_DT
data = mujoco.MjData(model)

NAMES = [model.actuator(i).name for i in range(model.nu)]
CTRL_RANGE = model.actuator_ctrlrange.copy()
GRIPPER = model.nu - 1                      # actuator8 = gripper tendon
ARM = list(range(model.nu - 1))             # actuators 1-7

# home arm configuration (from the model's "home" keyframe), gripper open
HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853*3])
HOME_CTRL = np.concatenate([HOME_ARM, [255.0]])


def reset_home():
    """Reset arm to the home pose and gripper open, leaving scene objects
    (e.g. the cube) at their XML default positions."""
    mujoco.mj_resetData(model, data)        # objects -> XML defaults, arm -> 0
    data.qpos[:7] = HOME_ARM
    data.qpos[7:9] = 0.04                    # both fingers open
    data.ctrl[:] = HOME_CTRL
    mujoco.mj_forward(model, data)


def reset_episode(sm, rng, viewer=None):
    """Swap in a freshly randomized episode: scene_gen recompiles the MjSpec
    into a new model/data pair (object + bin re-placed) AND samples a randomized
    initial arm config + gripper opening (already written to qpos). We mirror
    that start pose into ctrl so the position servos hold it instead of snapping
    back to home, then rebind the viewer scene buffers. The episode info dict
    (randomization params, incl. start pose) is returned for the recorder."""
    global model, data, _sim_accum, _last_step_t, _episode_start
    model, data, info = sm.new_episode(rng)
    model.opt.timestep = SIM_DT
    # scene_gen randomizes the start pose; fall back to home if absent (old data)
    start_arm = np.asarray(info.get("start_arm", HOME_ARM), float)
    start_grip_ctrl = float(info.get("start_grip_ctrl", 255.0))
    start_grip_qpos = float(info.get("start_grip", 0.04))
    data.ctrl[:7] = start_arm
    data.ctrl[GRIPPER] = start_grip_ctrl
    _episode_start = (start_arm.copy(), start_grip_qpos, start_grip_ctrl)
    mujoco.mj_forward(model, data)
    _sim_accum = 0.0
    _last_step_t = None
    if viewer is not None:
        viewer.rebind()
    return info


def home_arm_only():
    """Restore the arm + gripper to the current episode randomized start pose
    while leaving scene objects in place. Used by the R key during recording,
    where reset_home (mj_resetData) would teleport the object and bin back to
    their compiled defaults. Resetting to the episode start (not a fixed home)
    keeps recorded resets consistent with the start-pose randomization."""
    arm, grip_qpos, grip_ctrl = _episode_start
    data.qpos[:7] = arm
    data.qpos[7:9] = grip_qpos
    data.ctrl[:7] = arm
    data.ctrl[GRIPPER] = grip_ctrl
    mujoco.mj_forward(model, data)


def clamp():
    np.clip(data.ctrl, CTRL_RANGE[:, 0], CTRL_RANGE[:, 1], out=data.ctrl)


_sim_accum = 0.0
_last_step_t = None
# this episode start pose (arm[7], gripper qpos, gripper ctrl); set by
# reset_episode, restored by home_arm_only on the R key. Defaults to home.
_episode_start = (HOME_ARM.copy(), 0.04, 255.0)


def step_physics():
    """Advance physics in REAL TIME with a fixed-timestep accumulator.

    The old loop took one mj_step (SIM_DT) per rendered frame while the
    window is vsync-locked to ~60 fps, so the sim ran at ~0.12x realtime
    AND the controller (which advances targets by wall-clock dt) effectively
    commanded the arm ~8x too fast in sim time -> the held cube got flung.
    Here we step floor(frame_dt / SIM_DT) sub-steps so sim time tracks the
    wall clock; capped to avoid a spiral of death after a stall/pause."""
    global _sim_accum, _last_step_t
    now = time.time()
    frame_dt = 0.0 if _last_step_t is None else now - _last_step_t
    _last_step_t = now
    _sim_accum += min(frame_dt, 0.05)        # clamp huge frames (startup/pause)
    n = 0
    while _sim_accum >= SIM_DT and n < 50:
        mujoco.mj_step(model, data)
        _sim_accum -= SIM_DT
        n += 1


# ---------------------------------------------------------------- viewer
class Viewer:
    def __init__(self, title="Panda"):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed (no display?)")
        self.win = glfw.create_window(1200, 900, title, None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.cam.lookat[:] = [0.3, 0.0, 0.4]
        self.cam.distance = 2.2
        self.cam.azimuth = 120
        self.cam.elevation = -20

        self._lastx = self._lasty = 0.0
        self._btn_left = self._btn_right = self._btn_mid = False
        glfw.set_mouse_button_callback(self.win, self._mouse_button)
        glfw.set_cursor_pos_callback(self.win, self._cursor)
        glfw.set_scroll_callback(self.win, self._scroll)

        # picture-in-picture: C cycles through each camera in RGB then depth
        self._cam_names = [model.camera(i).name for i in range(model.ncam)]
        self._pip_views = ([(n, "rgb") for n in self._cam_names] +
                           [(n, "depth") for n in self._cam_names])
        self._pip_idx = (self._pip_views.index(("wrist", "rgb"))
                         if ("wrist", "rgb") in self._pip_views else 0)
        self._pip = len(self._pip_views) > 0
        self._c_was = self._p_was = False
        if self._pip_views:
            self.pip_scene = mujoco.MjvScene(model, maxgeom=10000)
            self.pip_cam = mujoco.MjvCamera()
            self.pip_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # OpenGL z-buffer -> metric depth (nonlinear): z = zn / (1 - d(1 - zn/zf))
        self._znear = model.vis.map.znear * model.stat.extent
        self._zfar = model.vis.map.zfar * model.stat.extent

        # coordinate-frame axes (X=red, Y=green, Z=blue); F cycles which frames
        self._frames = [(mujoco.mjtFrame.mjFRAME_WORLD, "world"),
                        (mujoco.mjtFrame.mjFRAME_SITE, "sites (incl. grip)"),
                        (mujoco.mjtFrame.mjFRAME_BODY, "bodies"),
                        (mujoco.mjtFrame.mjFRAME_NONE, "off")]
        self._frame_idx = 0
        self._f_was = False
        self._g_was = False
        self.opt.frame = self._frames[self._frame_idx][0]
        self._depth_tex = None   # lazy GL texture for the depth-PiP blit

        # grip force HUD: live clamp-force gauge (V toggles, default on)
        self._force_hud = True
        self._v_was = False
        self._finger_geoms = {gid for gid in range(model.ngeom)
                              if model.body(model.geom_bodyid[gid]).name
                              in ("left_finger", "right_finger")}

        # ---- data-collection offscreen grab (reuses THIS GL context) ----
        # Clean RGB needs its own scene + option (no frame axes / no HUD) and a
        # fixed camera; rendered into the offscreen framebuffer and read back.
        self.grab_scene = mujoco.MjvScene(model, maxgeom=10000)
        self.opt_clean = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt_clean)
        self.grab_cam = mujoco.MjvCamera()
        self.grab_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def rebind(self):
        """Rebuild scene buffers after the global model object is replaced by a
        new episode (scene_gen recompiles the MjSpec into a fresh model).
        Topology is held constant across episodes (same geom/mesh counts), so
        the GL render context self.ctx stays valid; only the MjvScene objects
        and model-derived caches need refreshing."""
        global model, data
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.grab_scene = mujoco.MjvScene(model, maxgeom=10000)
        if self._pip_views:
            self.pip_scene = mujoco.MjvScene(model, maxgeom=10000)
        self._cam_names = [model.camera(i).name for i in range(model.ncam)]
        self._znear = model.vis.map.znear * model.stat.extent
        self._zfar = model.vis.map.zfar * model.stat.extent
        self._finger_geoms = {gid for gid in range(model.ngeom)
                              if model.body(model.geom_bodyid[gid]).name
                              in ("left_finger", "right_finger")}

    def grab(self, cam_name, w=640, h=480):
        """Render one clean RGB frame from a named fixed camera into the
        offscreen buffer and return it as an (h, w, 3) uint8 array. Reuses the
        live GL context so no separate mujoco.Renderer is needed."""
        global model, data
        self.grab_cam.fixedcamid = self._cam_names.index(cam_name)
        mujoco.mjv_updateScene(model, data, self.opt_clean, None,
                               self.grab_cam, mujoco.mjtCatBit.mjCAT_ALL,
                               self.grab_scene)
        rect = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mujoco.mjr_render(rect, self.grab_scene, self.ctx)
        rgb = np.zeros(w * h * 3, np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.ctx)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        # GL origin is bottom-left; flip vertically to image (top-left) order.
        return np.ascontiguousarray(rgb.reshape(h, w, 3)[::-1])

    def grab_rgbd(self, cam_name, w=640, h=480):
        """Render RGB + metric depth from a named fixed camera in one pass.

        Returns (rgb, depth):
            rgb   (h, w, 3) uint8   -- same content as grab()
            depth (h, w)    float32 -- linear metric depth in metres, top-left
                                       origin (matches rgb). Background / sky
                                       reads ~= self._zfar (nothing hit).

        Piggybacks on grab()`s offscreen setup so the extra cost is just one
        mjr_readPixels including the depth buffer + the z-buffer -> metres
        math (identical formula to _draw_depth). Kept as a separate method so
        callers that only need rgb pay no cost."""
        global model, data
        self.grab_cam.fixedcamid = self._cam_names.index(cam_name)
        mujoco.mjv_updateScene(model, data, self.opt_clean, None,
                               self.grab_cam, mujoco.mjtCatBit.mjCAT_ALL,
                               self.grab_scene)
        rect = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mujoco.mjr_render(rect, self.grab_scene, self.ctx)
        rgb = np.zeros(w * h * 3, np.uint8)
        depth_raw = np.zeros(w * h, np.float32)
        mujoco.mjr_readPixels(rgb, depth_raw, rect, self.ctx)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        # OpenGL depth buffer stores nonlinear d in [0,1]. Convert to metres:
        #   z = zn / (1 - d * (1 - zn/zf))
        # d=1 (nothing hit / far plane) -> z = zf; d=0 (near plane) -> z = zn.
        zn, zf = self._znear, self._zfar
        z_metric = zn / (1.0 - depth_raw * (1.0 - zn / zf))
        # Flip to top-left origin so shapes/orientations agree with rgb.
        rgb_img = np.ascontiguousarray(rgb.reshape(h, w, 3)[::-1])
        depth_img = np.ascontiguousarray(z_metric.reshape(h, w)[::-1]).astype(np.float32)
        return rgb_img, depth_img

    def _mouse_button(self, w, button, act, mods):
        self._btn_left = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._btn_right = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._btn_mid = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._lastx, self._lasty = glfw.get_cursor_pos(w)

    def _cursor(self, w, xpos, ypos):
        dx, dy = xpos - self._lastx, ypos - self._lasty
        self._lastx, self._lasty = xpos, ypos
        if not (self._btn_left or self._btn_right or self._btn_mid):
            return
        _, height = glfw.get_window_size(w)
        action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if (self._btn_right or self._btn_mid)
                  else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, self.scene, self.cam)

    def _scroll(self, w, xoff, yoff):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff,
                              self.scene, self.cam)

    def held(self, key):
        return glfw.get_key(self.win, key) == glfw.PRESS

    def running(self):
        return not glfw.window_should_close(self.win)

    def poll_view_keys(self):
        """C cycles the PiP camera + RGB/depth mode; P toggles the PiP on/off."""
        if not self._pip_views:
            return
        c = self.held(glfw.KEY_C)
        if c and not self._c_was:
            self._pip_idx = (self._pip_idx + 1) % len(self._pip_views)
            n, m = self._pip_views[self._pip_idx]
            print(f"PiP: {n} ({m})")
        self._c_was = c
        pk = self.held(glfw.KEY_P)
        if pk and not self._p_was:
            self._pip = not self._pip
            print(f"PiP: {'on' if self._pip else 'off'}")
        self._p_was = pk
        f = self.held(glfw.KEY_F)
        if f and not self._f_was:
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)
            self.opt.frame = self._frames[self._frame_idx][0]
            print(f"frame axes: {self._frames[self._frame_idx][1]}")
        self._f_was = f
        # G snaps the gripper fully closed (grasp) or fully open in one press,
        # guaranteeing a full clamp (holding [ can leave it half-closed -> slip)
        g = self.held(glfw.KEY_G)
        if g and not self._g_was:
            closing = data.ctrl[GRIPPER] >= 128.0   # currently open-ish -> close
            data.ctrl[GRIPPER] = 0.0 if closing else 255.0
            print(f"gripper: {'close' if closing else 'open'}")
        self._g_was = g
        # V toggles the live grip-force gauge
        vk = self.held(glfw.KEY_V)
        if vk and not self._v_was:
            self._force_hud = not self._force_hud
            print(f"force HUD: {'on' if self._force_hud else 'off'}")
        self._v_was = vk

    def _draw_depth(self, rect, w, h):
        """Read the GL depth buffer for `rect`, colorize (near=bright), blit it back.

        macOS uses a Metal-backed legacy GL 2.1 context where glDrawPixels is a
        silent no-op, so mjr_drawPixels does nothing. We upload the colorized
        depth as a texture and draw it on a fixed-function quad over the rect."""
        n = w * h
        rgb = np.zeros(n * 3, np.uint8)
        depth = np.zeros(n, np.float32)
        mujoco.mjr_readPixels(rgb, depth, rect, self.ctx)
        zn, zf = self._znear, self._zfar
        z = zn / (1.0 - depth * (1.0 - zn / zf))          # metric metres
        # auto-range per-frame so every camera shows useful contrast (the
        # arm-side wrist sees ~0.1-0.8 m while front sees ~1-8 m)
        valid = z[depth < 0.9999]
        if valid.size > 100:
            lo, hi = np.percentile(valid, [5, 95])
            hi = max(hi, lo + 1e-3)
        else:
            lo, hi = 0.1, 1.5
        zc = np.clip(z, lo, hi)
        g = ((1.0 - (zc - lo) / (hi - lo)) * 255).astype(np.uint8)   # near = bright
        vis = np.repeat(g[:, None], 3, axis=1).reshape(h, w, 3)  # row0 = bottom (GL order)

        if self._depth_tex is None:
            self._depth_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._depth_tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_VIEWPORT_BIT |
                        GL.GL_TEXTURE_BIT | GL.GL_CURRENT_BIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._depth_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, vis.tobytes())
        GL.glViewport(rect.left, rect.bottom, rect.width, rect.height)
        GL.glMatrixMode(GL.GL_PROJECTION); GL.glPushMatrix(); GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW); GL.glPushMatrix(); GL.glLoadIdentity()
        GL.glDisable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_TEXTURE_2D); GL.glColor3f(1, 1, 1)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(-1, -1)
        GL.glTexCoord2f(1, 0); GL.glVertex2f( 1, -1)
        GL.glTexCoord2f(1, 1); GL.glVertex2f( 1,  1)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(-1,  1)
        GL.glEnd()
        GL.glMatrixMode(GL.GL_PROJECTION); GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW); GL.glPopMatrix()
        GL.glPopAttrib()

    def _clamp_force(self):
        """Total contact normal force on whatever the fingers are squeezing (N)."""
        f6 = np.zeros(6)
        tot = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = c.geom1 in self._finger_geoms
            g2 = c.geom2 in self._finger_geoms
            if g1 != g2:                       # exactly one side is a finger -> grip
                mujoco.mj_contactForce(model, data, i, f6)
                tot += abs(f6[0])              # f6[0] = normal force in contact frame
        return tot

    @staticmethod
    def _force_color(frac):
        frac = max(0.0, min(1.0, frac))
        if frac < 0.5:                         # green -> yellow
            t = frac / 0.5
            return (0.2 + 0.7 * t, 0.85, 0.2)
        t = (frac - 0.5) / 0.5                  # yellow -> red
        return (0.9, 0.85 - 0.65 * t, 0.15)

    def _draw_force_bar(self, fw, fh):
        force = self._clamp_force()
        cmd = (1.0 - data.ctrl[GRIPPER] / 255.0) * 100.0   # ctrl=0 -> 100% closed
        fmax = 6.0
        frac = force / fmax
        bw, bh = 26, 260
        x0, y0 = 30, fh // 2 - bh // 2
        mujoco.mjr_rectangle(mujoco.MjrRect(x0 - 3, y0 - 3, bw + 6, bh + 6), 0, 0, 0, 1)
        mujoco.mjr_rectangle(mujoco.MjrRect(x0, y0, bw, bh), 0.15, 0.15, 0.18, 1)
        fill = int(bh * max(0.0, min(1.0, frac)))
        if fill > 0:
            r, g, b = self._force_color(frac)
            mujoco.mjr_rectangle(mujoco.MjrRect(x0, y0, bw, fill), r, g, b, 1)
        # white tick at the cube's weight (~0.49 N): above it = enough to hold
        ty = y0 + int(bh * (0.49 / fmax))
        mujoco.mjr_rectangle(mujoco.MjrRect(x0 - 6, ty, bw + 12, 2), 1, 1, 1, 1)
        rect = mujoco.MjrRect(x0 - 4, y0 + bh + 6, 220, 48)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           rect, "grip force\ncommand",
                           f"{force:.1f} N\n{cmd:.0f}% closed", self.ctx)

    def render(self, marker=None):
        """Draw the main view (+ optional target marker, PiP inset, force HUD)
        and present the frame. `marker` is the Cartesian target world position."""
        # Rebuild the abstract scene from the current physics state.
        mujoco.mjv_updateScene(model, data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if marker is not None:                       # draw target as a sphere
            # Append one transient geom past the scene's current count, fill it
            # in (a small green translucent sphere), then bump ngeom so the
            # renderer includes it. Rebuilt every frame, not persistent.
            g = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]),
                np.asarray(marker, float), np.eye(3).ravel(),
                np.array([0.1, 0.9, 0.2, 0.6], np.float32))
            self.scene.ngeom += 1
        fw, fh = glfw.get_framebuffer_size(self.win)
        full = mujoco.MjrRect(0, 0, fw, fh)
        mujoco.mjr_render(full, self.scene, self.ctx)
        # picture-in-picture, bottom-right corner with a black border
        if self._pip and self._pip_views:
            cam_name, mode = self._pip_views[self._pip_idx]
            self.pip_cam.fixedcamid = self._cam_names.index(cam_name)
            mujoco.mjv_updateScene(model, data, self.opt, None, self.pip_cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.pip_scene)
            pw, ph = fw // 4, fh // 4
            inner = mujoco.MjrRect(fw - pw - 12, 12, pw, ph)
            border = mujoco.MjrRect(inner.left - 2, inner.bottom - 2,
                                    inner.width + 4, inner.height + 4)
            mujoco.mjr_rectangle(border, 0, 0, 0, 1)
            mujoco.mjr_render(inner, self.pip_scene, self.ctx)
            if mode == "depth":
                self._draw_depth(inner, pw, ph)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, inner,
                               f"{cam_name} ({mode})", "[C] switch  [P] hide",
                               self.ctx)
        if self._force_hud:
            self._draw_force_bar(fw, fh)
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def close(self):
        glfw.terminate()


class RecordSession:
    """Episode lifecycle + frame capture for ACT data collection.

    Wraps a SceneManager (procedural scenes) and a Recorder (LeRobotDataset).
    Lifecycle keys are edge-detected each frame: ENTER saves the current take as
    a successful episode and starts a new one, BACKSPACE discards the take and
    retries with a fresh scene, END finalizes the dataset and stops. Frames are
    captured at a fixed rate (decimated against wall-clock dt) by grabbing every
    camera through the viewer offscreen buffer."""

    def __init__(self, sm, recorder, rng, cams, viewer, fps=30):
        self.sm = sm
        self.rec = recorder
        self.rng = rng
        self.cams = cams
        self.v = viewer
        self.fps = fps
        self._frame_dt = 1.0 / fps
        self._accum = 0.0
        self._meta_written = False
        self._ended = False
        self._enter_was = self._back_was = self._end_was = False

    def new_scene(self):
        """Begin a fresh randomized episode and start recording it."""
        info = reset_episode(self.sm, self.rng, self.v)
        self.rec.start_episode(info)
        self._accum = 0.0
        if not self._meta_written:
            self._write_camera_meta()
            self._meta_written = True
        print(f"[rec] recording (target episode #{self.rec.num_episodes + 1}) -- "
              f"ENTER save | BACKSPACE redo | END finish")
        return info

    def capture(self, dt):
        """Decimate to fps and buffer one (images, state, action, depths) timestep.

        RGB + metric depth are grabbed together via `Viewer.grab_rgbd` so we
        only do one offscreen render per camera per frame."""
        self._accum += dt
        if self._accum < self._frame_dt:
            return
        self._accum -= self._frame_dt
        imgs, depths = {}, {}
        for c in self.cams:
            rgb, dep = self.v.grab_rgbd(c)
            imgs[c] = rgb
            depths[c] = dep
        state = np.concatenate([data.qpos[:7], [data.qpos[7]]])
        action = np.concatenate([data.ctrl[:7], [data.ctrl[GRIPPER]]])
        self.rec.add(imgs, state, action, depths=depths)

    def poll_keys(self):
        """Edge-detect lifecycle keys -> 'success' | 'cancel' | 'end' | None."""
        ev = None
        e = self.v.held(glfw.KEY_ENTER)
        if e and not self._enter_was:
            ev = "success"
        self._enter_was = e
        b = self.v.held(glfw.KEY_BACKSPACE)
        if b and not self._back_was:
            ev = "cancel"
        self._back_was = b
        n = self.v.held(glfw.KEY_END)
        if n and not self._end_was:
            ev = "end"
        self._end_was = n
        return ev

    def handle(self, ev):
        """Apply success/cancel: persist or discard the take, then start a new
        scene. Returns True when a new scene was started (the caller may need to
        re-seed Cartesian targets). 'end' is handled by the caller directly."""
        if ev == "success":
            self.rec.success()
            self.new_scene()
            return True
        if ev == "cancel":
            self.rec.cancel()
            self.new_scene()
            return True
        return False

    def finish(self):
        """Finalize the dataset once (idempotent)."""
        if not self._ended:
            self.rec.end()
            self._ended = True

    def _write_camera_meta(self):
        """Dump camera intrinsics/extrinsics once per session for downstream
        calibration. Fixed cams (front, diag) get a static world pose; the
        wrist cam is eye-in-hand, so only its intrinsics are stored and the
        extrinsic is per-frame (derivable from the recorded joint state)."""
        meta = {"image_wh": [640, 480], "cameras": {}}
        for c in self.cams:
            cid = model.camera(c).id
            entry = {"fovy_deg": float(model.cam_fovy[cid])}
            if c == "wrist":
                entry["mount"] = "eye_in_hand"
            else:
                entry["mount"] = "fixed"
                entry["world_pos"] = data.cam_xpos[cid].tolist()
                entry["world_mat"] = data.cam_xmat[cid].reshape(3, 3).tolist()
            meta["cameras"][c] = entry
        path = self.rec.root / "camera_meta.json"
        path.write_text(json.dumps(meta, indent=2))
        print(f"[rec] wrote camera metadata -> {path}")


def print_actuators():
    print(f"\n{model.nu} actuators:")
    for i, n in enumerate(NAMES):
        lo, hi = CTRL_RANGE[i]
        print(f"  [{i+1}] {n:<12} [{lo:+.3f}, {hi:+.3f}]")


# ---------------------------------------------------------------- per-joint teleop
def run_joints(record=None):
    """Per-joint teleop: pick a joint (1-7 / TAB) and hold UP/DOWN to slew its
    position-servo target. `record` (when --record) is the
    (SceneManager, Recorder, rng, cams) tuple that turns the session into ACT
    data collection."""
    print_actuators()
    print(__doc__[__doc__.index("Per-joint teleop"):__doc__.index("The Panda")])
    v = Viewer("Panda joint teleop")
    session = RecordSession(*record, v) if record else None
    if session is not None:
        session.new_scene()                          # randomized scene + start recording
    else:
        reset_home()
    sel = 0                                          # currently selected arm joint
    paused = False
    tab_was = False                                  # edge-detect state for TAB
    last = time.time()

    while v.running():
        now = time.time(); dt = now - last; last = now   # wall-clock frame dt
        if v.held(glfw.KEY_ESCAPE):
            break
        v.poll_view_keys()                           # C/P/F/G/V viewer hotkeys
        if session is not None:
            ev = session.poll_keys()                 # ENTER/BACKSPACE/END -> event
            if ev == "end":
                session.finish()
                break
            session.handle(ev)                       # save/discard + next scene
        # number keys 1-7 select a joint directly (KEY_1 + i)
        for i in range(len(ARM)):
            if v.held(glfw.KEY_1 + i):
                sel = i
        # edge-detect TAB (fire once per press, not every frame held) to cycle
        tab = v.held(glfw.KEY_TAB)
        if tab and not tab_was:
            sel = (sel + 1) % len(ARM)
            print(f"selected [{sel+1}] {NAMES[sel]}")
        tab_was = tab

        if v.held(glfw.KEY_UP):
            data.ctrl[sel] += MOVE_RATE * dt
        if v.held(glfw.KEY_DOWN):
            data.ctrl[sel] -= MOVE_RATE * dt
        if v.held(glfw.KEY_RIGHT_BRACKET):
            data.ctrl[GRIPPER] += GRIP_RATE * dt
        if v.held(glfw.KEY_LEFT_BRACKET):
            data.ctrl[GRIPPER] -= GRIP_RATE * dt
        if v.held(glfw.KEY_R):
            home_arm_only() if session is not None else reset_home()
        if v.held(glfw.KEY_SPACE):
            paused = not paused
            time.sleep(0.15)

        clamp()
        if not paused:
            step_physics()
        v.render()
        if session is not None and not paused:
            session.capture(dt)
    if session is not None:
        session.finish()
    v.close()


# ---------------------------------------------------------------- scripted demo
def run_scripted():
    print_actuators()
    print("\nRunning scripted demo. ESC or close window to quit.")
    v = Viewer("Panda demo")
    reset_home()

    # (arm 7-vector target, gripper ctrl, duration s)
    waypoints = [
        (HOME_ARM, 255, 1.5),
        (np.array([0.0, 0.4, 0.0, -2.0, 0.0, 2.4, -0.785]), 255, 2.5),   # reach down/forward
        (np.array([0.0, 0.4, 0.0, -2.0, 0.0, 2.4, -0.785]), 0,   1.0),   # close gripper
        (np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, -0.785]), 0,  2.5),   # lift
        (np.array([1.2, -0.3, 0.0, -1.8, 0.0, 1.5, -0.785]), 0,  2.5),   # swing to side
        (np.array([1.2, 0.2, 0.0, -2.0, 0.0, 2.2, -0.785]), 255, 2.0),   # place + open
        (HOME_ARM, 255, 2.5),                                            # back home
    ]

    def move_to(arm_target, grip, duration):
        start = data.ctrl.copy()
        target = np.concatenate([arm_target, [grip]])
        t0 = time.time()
        while (t := time.time() - t0) < duration and v.running():
            if v.held(glfw.KEY_ESCAPE):
                return False
            a = min(1.0, t / duration)
            data.ctrl[:] = (1 - a) * start + a * target
            clamp()
            step_physics()
            v.render()
        return v.running()

    for arm, grip, dur in waypoints:
        if not move_to(arm, grip, dur):
            break
    while v.running() and not v.held(glfw.KEY_ESCAPE):
        step_physics()
        v.render()
    v.close()


# ---------------------------------------------------------------- cartesian teleop
def run_cartesian(record=None):
    """Drive the grip site in Cartesian space; IK solves arm joints per frame."""
    from ik import PandaIK

    ik = PandaIK(model)                              # share the loaded model
    print_actuators()
    print("""
Cartesian teleop controls:
  W / S      +X / -X   (forward / back)
  A / D      +Y / -Y   (left / right)
  Q / E      +Z / -Z   (up / down)
  U / O      roll  -/+        (about world X)
  I / K      pitch -/+        (about world Y)
  J / L      yaw   -/+        (about world Z)
  ] / [      open / close gripper (hold)
  G          toggle gripper fully closed / open (grasp)
  C          switch the picture-in-picture camera + RGB/depth mode
  P          toggle the picture-in-picture on/off
  F          cycle coordinate-frame axes (world -> sites -> bodies -> off)
  V          toggle the live grip-force gauge
  R          reset to home pose
  SPACE      pause / resume physics
  ESC        quit
The green sphere marks the commanded target; the corner inset is a live camera
(C switches which camera + RGB/depth; P hides it).""")

    v = Viewer("Panda cartesian teleop")
    session = RecordSession(*record, v) if record else None
    if session is not None:
        session.new_scene()
    else:
        reset_home()

    LIN_RATE = 0.20          # m/s
    ANG_RATE = 1.0           # rad/s
    tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
    tgt_pos = tgt_pos.copy()
    paused = False
    last = time.time()

    while v.running():
        now = time.time(); dt = now - last; last = now
        if v.held(glfw.KEY_ESCAPE):
            break
        v.poll_view_keys()
        if session is not None:
            ev = session.poll_keys()
            if ev == "end":
                session.finish()
                break
            if session.handle(ev):          # new scene -> re-seed the target
                tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
                tgt_pos = tgt_pos.copy()

        d = np.zeros(3)
        if v.held(glfw.KEY_W): d[0] += 1
        if v.held(glfw.KEY_S): d[0] -= 1
        if v.held(glfw.KEY_A): d[1] += 1
        if v.held(glfw.KEY_D): d[1] -= 1
        if v.held(glfw.KEY_Q): d[2] += 1
        if v.held(glfw.KEY_E): d[2] -= 1
        if np.any(d):
            tgt_pos = tgt_pos + LIN_RATE * dt * d

        w = np.zeros(3)
        if v.held(glfw.KEY_U): w[0] -= 1
        if v.held(glfw.KEY_O): w[0] += 1
        if v.held(glfw.KEY_I): w[1] -= 1
        if v.held(glfw.KEY_K): w[1] += 1
        if v.held(glfw.KEY_J): w[2] -= 1
        if v.held(glfw.KEY_L): w[2] += 1
        if np.any(w):
            # integrate the angular-velocity command into the target quaternion
            # (rotate by w*rate over dt), then renormalize to kill drift.
            mujoco.mju_quatIntegrate(tgt_quat, w * ANG_RATE, dt)
            mujoco.mju_normalize4(tgt_quat)

        if v.held(glfw.KEY_RIGHT_BRACKET):
            data.ctrl[GRIPPER] += GRIP_RATE * dt
        if v.held(glfw.KEY_LEFT_BRACKET):
            data.ctrl[GRIPPER] -= GRIP_RATE * dt
        if v.held(glfw.KEY_R):
            home_arm_only() if session is not None else reset_home()
            tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
            tgt_pos = tgt_pos.copy()
        if v.held(glfw.KEY_SPACE):
            paused = not paused
            time.sleep(0.15)

        # IK: solve arm joints that put the EE at (tgt_pos, tgt_quat), damped and
        # step-limited (max_dq) so the servos track smoothly without snapping.
        q = ik.solve(data.qpos[:7], tgt_pos, tgt_quat, iters=30, max_dq=0.08)
        data.ctrl[:7] = q
        clamp()                                      # keep ctrl within actuator limits
        if not paused:
            step_physics()                           # advance sim in real time
        v.render(marker=tgt_pos)
        if session is not None and not paused:
            session.capture(dt)                      # buffer a frame (decimated to fps)
    if session is not None:
        session.finish()                             # finalize dataset if recording
    v.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    record_on = "--record" in argv
    argv = [a for a in argv if a != "--record"]
    mode = argv[0] if argv else "joints"

    record = None
    if record_on:
        # lazy import: scene_gen + recorder pull in heavy deps (OMP, lerobot)
        from scene_gen import SceneManager
        import recorder as rec_mod
        sm = SceneManager()
        ds_root = os.path.join(HERE, "ACT", "data", "panda_pick_place")
        recorder = rec_mod.Recorder.open(ds_root, fps=30, resume=True)
        rng = np.random.default_rng()
        record = (sm, recorder, rng, rec_mod.CAMS)

    if mode == "joints":
        run_joints(record)
    elif mode == "scripted":
        if record_on:
            print("scripted mode does not support --record")
            sys.exit(1)
        run_scripted()
    elif mode in ("cartesian", "ik"):
        run_cartesian(record)
    else:
        print(f"unknown mode '{mode}' (use: joints | scripted | cartesian)")
        sys.exit(1)

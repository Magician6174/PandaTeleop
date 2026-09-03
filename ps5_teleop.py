"""PS5 DualSense teleop for the Franka Panda (MuJoCo), via pygame + IK.

Velocity (rate) control: stick/trigger deflection -> end-effector velocity,
integrated each frame into a target EE pose, solved by the shared PandaIK
(ik.py) and written to the position servos. This is the gamepad analogue of
control.py's `cartesian` mode -- control.py itself is left untouched.

Modes:
  teleop   - drive the Panda from the controller (default)
  probe    - read-test: print every axis/button/hat live so we can confirm
             this controller's real index map on THIS machine (SDL indices
             vary by OS/connection). Run this FIRST, then fix the map below.

Run (env python_robotics; macOS needs mjpython only if you switch to a passive
viewer -- this uses GLFW directly like control.py, so plain python is fine):
  python ps5_teleop.py probe     # confirm axis/button indices
  python ps5_teleop.py           # teleop

Control mapping (confirm indices with `probe`):
  Left stick            EE position X / Y
  L2 / R2 (analog)      EE position Z  down / up
  Right stick           EE rotation about X / Y
  L1 / R1               EE rotation about Z (roll) - / +
  Square / Circle (hold) gripper open / close
  Cross                 toggle grasp (full open <-> full close)
  Triangle              reset to home
  Options               pause / resume physics
  D-pad up / down       motion speed scale up / down
  D-pad right / left    (--record) save episode + next / discard + redo
  Create                (--record) finish + finalize the dataset
  R3 (right-stick click) optional clutch: motion only enabled while held
"""
import os
import sys
import time
import numpy as np
import mujoco

# Run SDL headless so pygame doesn't open its own window (GLFW owns the display).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

# Reuse the model, viewer, stepping and gripper plumbing from control.py.
# Importing runs control.py's module-level setup (loads the model) but NOT its
# __main__ block, so no teleop loop starts.
import control
from control import (model, data, Viewer, step_physics, reset_home, clamp,
                     GRIPPER, GRIP_RATE)
from ik import PandaIK

# ---------------------------------------------------------------- controller map
# VERIFY THESE with `python ps5_teleop.py probe`, then edit to match.
# Typical SDL/pygame DualSense layout (varies by OS + USB/BT):
AX_LX = 0      # left stick  X
AX_LY = 1      # left stick  Y
AX_RX = 2      # right stick X
AX_RY = 3      # right stick Y
AX_L2 = 4      # left trigger  (analog)
AX_R2 = 5      # right trigger (analog)

BTN_CROSS    = 0   # toggle grasp
BTN_CIRCLE   = 1   # gripper close (hold)
BTN_TRIANGLE = 3   # reset home
BTN_SQUARE   = 2   # gripper open (hold)
BTN_L1       = 9   # roll -
BTN_R1       = 10  # roll +s
BTN_CREATE   = 4   # finish recording (--record)
BTN_OPTIONS  = 6   # pause / resume
BTN_R3       = 8  # clutch (optional)
# D-pad: this DualSense reports it as BUTTONS (hats=0 on macOS/SDL2), confirmed
# via `probe`. Up/Down/Left/Right = 11/12/13/14.
BTN_DPAD_UP    = 11
BTN_DPAD_DOWN  = 12
BTN_DPAD_LEFT  = 13
BTN_DPAD_RIGHT = 14

# Triggers on SDL usually rest at -1.0 and read +1.0 fully pressed. If `probe`
# shows them resting near 0.0 instead, set TRIGGER_REST = 0.0.
TRIGGER_REST = -1.0

# Sign flips: set to -1 if `probe`/live test shows an axis moving the wrong way.
SIGN_LX, SIGN_LY = -1.0, -1.0   # stick-right -> -Y, stick-up -> +X
SIGN_RX, SIGN_RY = +1.0, +1.0

# ---------------------------------------------------------------- tuning
DEADZONE = 0.12         # ignore stick noise/drift within this of center
LIN_RATE = 0.20         # m/s   at full stick (matches control.py cartesian)
ANG_RATE = 1.0          # rad/s at full stick
SCALE_MIN, SCALE_MAX = 0.25, 2.0   # D-pad speed-scale clamp
SCALE_STEP = 1.25                  # multiply/divide per D-pad press


def deadzone(v, dz=DEADZONE):
    """Zero out small magnitudes, then rescale so motion starts smoothly from 0
    at the deadzone edge (no velocity jump at the threshold)."""
    if abs(v) < dz:
        return 0.0
    s = (abs(v) - dz) / (1.0 - dz)
    return np.sign(v) * s


def trig01(raw):
    """Map a trigger axis to 0..1 (released..fully pressed)."""
    t = (raw - TRIGGER_REST) / (1.0 - TRIGGER_REST)
    return float(np.clip(t, 0.0, 1.0))


def open_controller():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller found. Connect the DualSense (USB-C) and retry.")
        sys.exit(1)
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}  "
          f"(axes={js.get_numaxes()} buttons={js.get_numbuttons()} hats={js.get_numhats()})")
    return js


# ---------------------------------------------------------------- probe (read-test)
def run_probe():
    js = open_controller()
    print("\nMove every stick, squeeze each trigger, press each button / D-pad.")
    print("Note the index that lights up for each control, then edit the map at")
    print("the top of this file. Ctrl-C to quit.\n")
    na, nb, nh = js.get_numaxes(), js.get_numbuttons(), js.get_numhats()
    try:
        while True:
            pygame.event.pump()
            axes = [f"a{i}:{js.get_axis(i):+.2f}" for i in range(na)]
            btns = [f"b{i}" for i in range(nb) if js.get_button(i)]
            hats = [f"h{i}:{js.get_hat(i)}" for i in range(nh)]
            line = "  ".join(axes) + "   | " + (" ".join(btns) if btns else "-")
            if hats:
                line += "   | " + " ".join(hats)
            print("\r" + line.ljust(120), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")
    finally:
        pygame.quit()


# ---------------------------------------------------------------- teleop
def run_teleop(record=None):
    # `data`/`model` are imported by value; reset_episode swaps control's globals,
    # so re-grab the local names after any scene change (the recorder/grab path
    # reads control's globals directly and is unaffected).
    global model, data
    js = open_controller()
    ik = PandaIK(model)
    nb = js.get_numbuttons()
    nh = js.get_numhats()

    print(__doc__[__doc__.index("Control mapping"):])
    v = Viewer("Panda PS5 teleop")
    session = control.RecordSession(*record, v) if record else None
    if session is not None:
        session.new_scene()
        model, data = control.model, control.data
    else:
        reset_home()

    tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
    tgt_pos = tgt_pos.copy()
    scale = 1.0
    paused = False

    # Edge-detection state: buttons are polled level (held = True every frame),
    # so for one-shot actions we remember last frame and fire only on the
    # False->True transition. The D-pad is buttons 11-14 on this controller.
    prev = {b: False for b in (BTN_CROSS, BTN_TRIANGLE, BTN_OPTIONS, BTN_CREATE,
                               BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT,
                               BTN_DPAD_RIGHT)}
    last = time.time()

    def pressed(btn):
        return btn < nb and js.get_button(btn)

    while v.running():
        now = time.time(); dt = now - last; last = now
        pygame.event.pump()
        v.poll_view_keys()        # keep C/P/F/G/V viewer hotkeys working

        # Clutch disabled by default -> motion always enabled. To require holding
        # R3 as a deadman, swap to: enabled = pressed(BTN_R3)
        enabled = True

        # ---- position: left stick X/Y + triggers Z ----
        d = np.zeros(3)
        if enabled:
            lx = SIGN_LX * deadzone(js.get_axis(AX_LX))
            ly = SIGN_LY * deadzone(js.get_axis(AX_LY))
            d[0] = ly                       # stick up   -> +X (forward)
            d[1] = lx                       # stick right-> -Y handled by SIGN_LX
            z = trig01(js.get_axis(AX_L2)) - trig01(js.get_axis(AX_R2))
            d[2] = z                        # R2 up, L2 down
        # rate control: stick deflection d (unit-ish) -> EE velocity, integrated
        # into the target position this frame.
        if np.any(d):
            tgt_pos = tgt_pos + LIN_RATE * scale * dt * d

        # ---- orientation: right stick (about X/Y) + L1/R1 (about Z) ----
        w = np.zeros(3)
        if enabled:
            rx = SIGN_RX * deadzone(js.get_axis(AX_RX))
            ry = SIGN_RY * deadzone(js.get_axis(AX_RY))
            w[0] = rx                       # right stick X -> rotate about X
            w[1] = ry                       # right stick Y -> rotate about Y
            if pressed(BTN_R1): w[2] += 1.0
            if pressed(BTN_L1): w[2] -= 1.0
        if np.any(w):
            # integrate angular-velocity command into the target quaternion over
            # dt (scale = D-pad speed multiplier), then renormalize.
            mujoco.mju_quatIntegrate(tgt_quat, w * ANG_RATE * scale, dt)
            mujoco.mju_normalize4(tgt_quat)

        # ---- gripper: Square open / Circle close (hold), Cross toggles grasp ----
        if pressed(BTN_SQUARE):
            data.ctrl[GRIPPER] += GRIP_RATE * dt
        if pressed(BTN_CIRCLE):
            data.ctrl[GRIPPER] -= GRIP_RATE * dt
        if pressed(BTN_CROSS) and not prev[BTN_CROSS]:
            closing = data.ctrl[GRIPPER] >= 128.0
            data.ctrl[GRIPPER] = 0.0 if closing else 255.0
            print(f"gripper: {'close' if closing else 'open'}")
        prev[BTN_CROSS] = pressed(BTN_CROSS)

        # ---- utility presses ----
        if pressed(BTN_TRIANGLE) and not prev[BTN_TRIANGLE]:
            control.home_arm_only() if session is not None else reset_home()
            tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
            tgt_pos = tgt_pos.copy()
            print("reset home")
        prev[BTN_TRIANGLE] = pressed(BTN_TRIANGLE)

        if pressed(BTN_OPTIONS) and not prev[BTN_OPTIONS]:
            paused = not paused
            print(f"{'paused' if paused else 'resumed'}")
        prev[BTN_OPTIONS] = pressed(BTN_OPTIONS)

        # ---- recording lifecycle: D-pad right=save+next, left=redo, Create=finish ----
        if session is not None:
            ev = None
            if pressed(BTN_DPAD_RIGHT) and not prev[BTN_DPAD_RIGHT]:
                ev = "success"
            elif pressed(BTN_DPAD_LEFT) and not prev[BTN_DPAD_LEFT]:
                ev = "cancel"
            prev[BTN_DPAD_RIGHT] = pressed(BTN_DPAD_RIGHT)
            prev[BTN_DPAD_LEFT] = pressed(BTN_DPAD_LEFT)
            if pressed(BTN_CREATE) and not prev[BTN_CREATE]:
                ev = "end"
            prev[BTN_CREATE] = pressed(BTN_CREATE)
            if ev == "end":
                session.finish()
                break
            if ev and session.handle(ev):      # new scene -> re-grab + re-seed
                model, data = control.model, control.data
                tgt_pos, tgt_quat = ik.ee_pose(data.qpos[:7])
                tgt_pos = tgt_pos.copy()

        # ---- D-pad up/down: speed scale ----
        if pressed(BTN_DPAD_UP) and not prev[BTN_DPAD_UP]:
            scale = min(SCALE_MAX, scale * SCALE_STEP)
            print(f"speed scale: {scale:.2f}x")
        elif pressed(BTN_DPAD_DOWN) and not prev[BTN_DPAD_DOWN]:
            scale = max(SCALE_MIN, scale / SCALE_STEP)
            print(f"speed scale: {scale:.2f}x")
        prev[BTN_DPAD_UP] = pressed(BTN_DPAD_UP)
        prev[BTN_DPAD_DOWN] = pressed(BTN_DPAD_DOWN)

        # ---- IK + step + render (identical to control.py cartesian) ----
        q = ik.solve(data.qpos[:7], tgt_pos, tgt_quat, iters=30, max_dq=0.08)
        data.ctrl[:7] = q
        clamp()
        if not paused:
            step_physics()
        v.render(marker=tgt_pos)
        if session is not None and not paused:
            session.capture(dt)

    if session is not None:
        session.finish()
    v.close()
    pygame.quit()


if __name__ == "__main__":
    argv = sys.argv[1:]
    record_on = "--record" in argv
    argv = [a for a in argv if a != "--record"]
    mode = argv[0] if argv else "teleop"

    record = None
    if record_on:
        from scene_gen import SceneManager
        import recorder as rec_mod
        sm = SceneManager()
        ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ACT", "data", "panda_pick_place")
        rec = rec_mod.Recorder.open(ds_root, fps=30, resume=True)
        rng = np.random.default_rng()
        record = (sm, rec, rng, rec_mod.CAMS)

    if mode == "probe":
        if record_on:
            print("probe mode does not support --record")
            sys.exit(1)
        run_probe()
    elif mode == "teleop":
        run_teleop(record)
    else:
        print(f"unknown mode '{mode}' (use: teleop | probe)")
        sys.exit(1)

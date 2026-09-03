# Franka Emika Panda — MuJoCo simulation

Self-contained copy of the `franka_emika_panda` model from `mujoco_menagerie`
(7-DOF arm + parallel gripper), with a TCP `site name="grip"` added to the hand
and a small cube in the scene for future grasping.

## Files
- `panda.xml`     — robot model (copied from menagerie; added `grip` site + grasp tuning)
- `scene.xml`     — floor, lights, cube; includes `panda.xml`
- `assets/`       — meshes
- `control.py`    — GLFW viewer + per-joint / cartesian teleop + scripted demo
- `ik.py`         — MuJoCo-native DLS Jacobian IK (PandaIK; pos+orientation)
- `ps5_teleop.py` — PS5 DualSense teleop (velocity control via pygame + PandaIK)

## Run (conda env `python_robotics`, plain python — no mjpython needed)
```
python control.py            # per-joint teleop (default)
python control.py cartesian  # cartesian-space teleop (IK)
python control.py scripted   # scripted pick-and-place demo
python ps5_teleop.py probe   # PS5: print live axis/button indices
python ps5_teleop.py         # PS5: teleop the arm
python ik.py test            # headless IK accuracy check
```

## Teleop controls
`1`-`7` select arm joint · `TAB` cycle · `UP`/`DOWN` move · `]`/`[` gripper
open/close · `R` reset home · `SPACE` pause · mouse orbit/pan/zoom · `ESC` quit

## Notes
- Arm actuators 1-7 are position servos (`ctrl` = target joint angle, rad).
- Actuator 8 drives the gripper tendon over ctrl range `0-255` (255 = open).
- Home pose: `[0, 0, 0, -1.571, 0, 1.571, -0.785]`, gripper open.
- 7-DOF is redundant for a 6-DOF pose, so full position+orientation IK is
  achievable (Phase 2 — recommend MuJoCo-native DLS Jacobian IK, no URDF needed).

## Grasp tuning (why panda.xml differs from stock menagerie)

Two contact-solver tweaks were added to hold a grasped object steadily. Neither
is a friction or grip-force problem — measured grip force is ~7.5 N against the
0.49 N cube (15x margin), and gripping 3x harder changed nothing. Both fixes are
in the *contact solver*, and they address two distinct symptoms:

**1. Object slowly slid out and fell** — `<option>` line:
```xml
<option integrator="implicitfast" cone="elliptic" impratio="50"/>
```
MuJoCo friction is a soft (regularized) constraint. With the default
`impratio=1` the friction constraint is as compliant as the normal constraint,
so under sustained gravity the object creeps downward at a slow constant
velocity even far below the friction limit. `impratio=50` makes friction ~50x
stiffer than normal contact (creep arrested); `cone="elliptic"` uses the
accurate elliptic friction cone instead of the default pyramidal one (more
stable for grasping). Measured: continuous creep -> holds indefinitely.

**2. Object shifted in the grip when rotating the EEF** — fingertip pads:
```xml
<geom ... condim="6" friction="1 1.0 0.1" priority="2" solref="0.005 1" .../>
```
A different softness: the contact's normal-direction springiness (`solref`).
The default time-constant (0.02) makes the pads behave like soft rubber, so when
the grip reorients the object elastically sinks a few mm into the give before
re-settling (not sliding — squishing). Shortening `solref` to `0.005 1` makes
the pads ~4x stiffer (hard-plastic-like). Only the pad geoms are changed; their
`priority="2"` means the grip contact uses their settings, so the floor and
other contacts stay soft + numerically stable. Measured per-axis in-grip shift
(over a continuous rotation): Z-spin 4.5 -> 0.3 mm, X-tilt 4.5 -> 0.65 mm.

**Known residual:** Y-axis tilt still shifts ~11 mm and is *not* solver-related
(grip force and rotation speed make no difference). It is geometric — the small
fingertip pads (~8.5 mm) let the 5 cm cube rock on its contact patches. Fix is
geometric: use a smaller cube (~2-3 cm) or enlarge the pad collision geoms.

`SIM_DT=0.001`: keep `solref` time-constant >= ~2x timestep (0.002). `0.005` is
a comfortable margin; `0.002` is the hard minimum and risks instability.

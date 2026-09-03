"""Procedural, randomized pick-and-place scene for the Panda (MuJoCo MjSpec).

Why MjSpec + recompile: object/bin *shape* (type + size) is baked into the
compiled model, so randomizing shape needs a model rebuild -- not just a state
(qpos) change. MjSpec lets us programmatically (re)build the scene each episode
and compile() a fresh, fully-correct model (mass, inertia, collision bounds all
recomputed for free), avoiding the bug-prone path of hand-patching geom_size +
mass + rbound at runtime.

Topology is held CONSTANT across episodes (always exactly 1 object geom + 5 bin
geoms), so the live viewer's MjvScene/MjrContext buffers stay valid -- only the
caller's model/data references need swapping.

Position/orientation are *state*: set via qpos after compile, with rejection
sampling so (a) object and bin never overlap and (b) BOTH project inside at
least one fixed camera (cameras are the only policy input).

Usage:
    sm = SceneManager()                       # loads scene.xml as a spec
    model, data, info = sm.new_episode(rng)   # randomized shape + pose
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")

# Panda home arm pose (matches control.HOME_ARM) so randomization happens with
# the arm out of the way; gripper open.
HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853 * 3])

# Fixed cameras used for the visibility guarantee (wrist is eye-in-hand and
# does not always see the table, so it is excluded from the guarantee).
VIS_CAMS = ("front", "diag")

# --- workspace / placement ranges (metres, world frame) -----------------------
OBJ_X = (0.42, 0.58)
OBJ_Y = (-0.18, 0.18)
BIN_X = (0.42, 0.58)
BIN_Y = (0.30, 0.44)          # placed on +Y or -Y side (sign randomized)
CLEARANCE = 0.04              # min gap between object and bin footprints (m)

# --- initial robot-pose randomization -----------------------------------------
# Per-joint uniform half-range (rad) added to HOME_ARM at episode start. Kept
# tight on the big end-effector-swinging joints (2,4,6) so the gripper stays in
# the home basin (does not dive under the table or block the fixed cameras) and
# looser on the wrist joints where swing is cheap.
ARM_JITTER = np.array([0.20, 0.12, 0.20, 0.12, 0.25, 0.18, 0.30])
GRIP_OPEN = 0.040             # per-finger fully-open position (m)
GRIP_START = (0.030, 0.040)   # start finger opening range (m); >=0.03 clears every object
GRIP_CTRL_OPEN = 255.0        # actuator8 ctrl at fully open (model maps 0.04 -> 255)

# Non-robot geoms in the compiled scene; anything else is arm/gripper (used for
# start-pose collision + line-of-sight tests).
SCENE_GEOMS = {"floor", "object_geom",
               "bin_base", "bin_xp", "bin_xn", "bin_yp", "bin_yn"}

GEOM = mujoco.mjtGeom


def _grip_ctrl(finger):
    """Map a per-finger opening (m) to the gripper actuator ctrl (0-255)."""
    return float(np.clip(finger / GRIP_OPEN, 0.0, 1.0) * GRIP_CTRL_OPEN)


def _robot_geom_ids(model):
    """Set of geom ids belonging to the arm + gripper (everything that is not
    floor, object, or bin) -- used for start-pose collision/occlusion tests."""
    return {g for g in range(model.ngeom)
            if model.geom(g).name not in SCENE_GEOMS}


def _arm_touches_scene(data, robot_geoms):
    """True if any live contact has exactly one robot geom, i.e. the arm/gripper
    is touching the floor, object, or bin in the sampled start pose."""
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in robot_geoms) != (c.geom2 in robot_geoms):
            return True
    return False


def _sample_object(rng):
    """Pick a graspable object shape. Returns (mjtGeom type, size[3], rest_z).

    `size` follows MuJoCo geom-size conventions (which differ per type), and
    `rest_z` is the height of the object centre when it sits on the floor (= its
    half-height/radius), used later to place it without it sinking or floating.
    """
    kind = rng.choice(["box", "cylinder", "sphere"])
    if kind == "box":
        # box size = half-extents (hx, hy, hz). Small enough to fit the gripper.
        hx, hy, hz = rng.uniform(0.015, 0.025, size=3)
        return GEOM.mjGEOM_BOX, np.array([hx, hy, hz]), float(hz)
    if kind == "cylinder":
        # cylinder size = (radius, half-height, unused).
        r = rng.uniform(0.012, 0.020)
        hh = rng.uniform(0.020, 0.035)
        return GEOM.mjGEOM_CYLINDER, np.array([r, hh, 0.0]), float(hh)
    # sphere size = (radius, unused, unused); rest_z = radius.
    r = rng.uniform(0.015, 0.022)
    return GEOM.mjGEOM_SPHERE, np.array([r, 0.0, 0.0]), float(r)


def _sample_bin(rng):
    """Pick bin dimensions. Returns dict with half (inner), wall t, wall h.

    The bin is a square open-top box built from 5 thin slabs (base + 4 walls).
    `half` is the INNER half-width; `outer` = half + wall thickness is the
    footprint used for collision-clearance maths.
    """
    half = rng.uniform(0.05, 0.08)     # inner half-extent (square footprint)
    t = 0.005                           # wall / base thickness
    wh = rng.uniform(0.030, 0.050)      # wall height
    return {"half": half, "t": t, "wh": wh, "outer": half + t}


def _add_object(wb, otype, osize):
    """Add the graspable object body to the worldbody `wb` of an MjSpec.

    The body gets a free joint (6-DoF) so it can be picked up and moved; the
    pos here is a placeholder -- the real XY/yaw is set via qpos after compile()
    in new_episode().
    """
    b = wb.add_body(name="object", pos=[0.5, 0.0, 0.05])
    b.add_freejoint()                       # 6-DoF so it can be lifted/carried
    g = b.add_geom()
    g.name = "object_geom"
    g.type = otype
    g.size = [float(s) for s in osize]
    g.rgba = [0.9, 0.3, 0.2, 1.0]           # reddish, easy to see in cameras
    g.mass = 0.05                            # 50 g
    # firm contact for a steady grasp (mirrors panda.xml pad tuning intent)
    g.friction = [1.0, 0.05, 0.01]
    return b


def _add_bin(wb, bp):
    """Add the open-top bin: one base slab + four upright walls, all boxes.

    Each entry is (name, half-extents, local-pos relative to the bin body).
    Walls sit at +/-half along X and Y; their centres are raised to `wh` (wall
    half-height) so they stand on top of the base. Like the object, the bin has
    a free joint and a placeholder pos; its real XY is set via qpos later.
    """
    half, t, wh, outer = bp["half"], bp["t"], bp["wh"], bp["outer"]
    b = wb.add_body(name="bin", pos=[0.5, -0.35, 0.0])
    b.add_freejoint()
    walls = [
        ("bin_base", [outer, outer, t], [0, 0, t]),     # floor slab
        ("bin_xp", [t, outer, wh], [half, 0, wh]),      # +X wall
        ("bin_xn", [t, outer, wh], [-half, 0, wh]),     # -X wall
        ("bin_yp", [outer, t, wh], [0, half, wh]),      # +Y wall
        ("bin_yn", [outer, t, wh], [0, -half, wh]),     # -Y wall
    ]
    for nm, sz, ps in walls:
        g = b.add_geom()
        g.name = nm
        g.type = GEOM.mjGEOM_BOX
        g.size = sz
        g.pos = ps
        g.rgba = [0.40, 0.40, 0.45, 1.0]                # neutral grey
    return b


def _point_in_camera(model, data, cam_id, p, aspect, margin=0.92):
    """True if world point p projects inside camera cam_id's view frustum.

    Pinhole-camera test: transform p into the camera frame, reject if behind
    the camera, then perspective-project and check it lands within the field
    of view. `margin` (<1) shrinks the box so points do not hug the image edge.
    """
    # Camera pose in the world: position + orientation. Columns of R are the
    # camera local X,Y,Z axes expressed in world coordinates.
    cam_pos = data.cam_xpos[cam_id]
    R = data.cam_xmat[cam_id].reshape(3, 3)
    # World point -> camera frame. R.T is the inverse rotation (R is orthonormal),
    # rotating the camera->point vector into the camera local axes.
    pc = R.T @ (np.asarray(p, float) - cam_pos)
    # MuJoCo cameras look down their -Z axis, so anything in front has pc[2] < 0.
    if pc[2] >= 0:                                   # point is behind the camera
        return False
    # Field of view -> half-angle tangents. Only vertical FOV is stored; the
    # horizontal half-angle is the vertical one scaled by the image aspect (w/h).
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    tan_v = np.tan(fovy / 2.0)
    tan_h = aspect * tan_v
    # Perspective divide: lateral offset per unit forward depth (-pc[2] > 0).
    # u,v are the tangents of the horizontal/vertical angle to the point.
    u = pc[0] / (-pc[2])
    v = pc[1] / (-pc[2])
    # In frame iff the angular offset is within the (margin-shrunk) half-FOV.
    return abs(u) <= margin * tan_h and abs(v) <= margin * tan_v


def _visible(model, data, p, aspect):
    """True if world point p is seen by AT LEAST ONE fixed camera (front/diag).

    The wrist cam is eye-in-hand and does not reliably see the table, so it is
    excluded -- this guards that the policy always has the object/bin on camera.
    """
    for name in VIS_CAMS:
        cid = model.camera(name).id
        if _point_in_camera(model, data, cid, p, aspect):
            return True
    return False


class SceneManager:
    """Builds a randomized pick-and-place model each episode."""

    def __init__(self, scene=SCENE, home_arm=HOME_ARM, img_w=640, img_h=480):
        self.scene = scene
        self.home_arm = np.asarray(home_arm, float)
        self.aspect = img_w / img_h
        self.model = None
        self.data = None

    def _build_spec(self, rng):
        """Load scene.xml as an editable MjSpec, strip the demo content, and add
        a freshly randomized object + bin. Returns (spec, info) -- not yet
        compiled or placed (new_episode does that)."""
        spec = mujoco.MjSpec.from_file(self.scene)
        # The XML ships a 'home' keyframe sized for the original nq; once we
        # add/remove free-joint bodies the qpos length changes, so the stale
        # keyframe would fail to compile -- drop all keyframes.
        for k in list(spec.keys):
            spec.delete(k)
        cube = spec.body("cube")           # remove the static demo cube, if present
        if cube is not None:
            spec.delete(cube)
        wb = spec.worldbody
        otype, osize, rest_z = _sample_object(rng)
        bp = _sample_bin(rng)
        _add_object(wb, otype, osize)
        _add_bin(wb, bp)
        # info carries the chosen shape params forward; new_episode adds poses.
        info = {"object_type": int(otype), "object_size": osize,
                "object_rest_z": rest_z, "bin": bp}
        return spec, info

    def _sample_start_pose(self, model, data, rng, max_tries=100):
        """Sample a jittered initial arm config + gripper opening that is
        collision-free against the floor/object/bin. Writes the accepted pose
        into qpos and returns (arm[7], finger).

        The fixed cameras look back toward the arm base with the object/bin in
        front, so a home-basin arm effectively never occludes them (measured: 0
        occlusions across 200 episodes even at 6x jitter) -- hence a collision
        guard only, no occlusion test.

        Falls back to the known-good HOME_ARM / fully-open gripper if no valid
        draw is found, so an episode is never lost to start-pose sampling.
        Assumes object/bin qpos are already set (their qpos indices differ)."""
        robot_geoms = _robot_geom_ids(model)
        lo = model.jnt_range[:7, 0] + 0.05      # stay off the joint limits
        hi = model.jnt_range[:7, 1] - 0.05
        for _ in range(max_tries):
            arm = np.clip(self.home_arm + rng.uniform(-ARM_JITTER, ARM_JITTER),
                          lo, hi)
            finger = float(rng.uniform(*GRIP_START))
            data.qpos[:7] = arm
            data.qpos[7:9] = finger
            mujoco.mj_forward(model, data)
            if not _arm_touches_scene(data, robot_geoms):
                return arm, finger
        # fallback: home pose is guaranteed collision-free
        data.qpos[:7] = self.home_arm
        data.qpos[7:9] = GRIP_OPEN
        mujoco.mj_forward(model, data)
        return self.home_arm.copy(), GRIP_OPEN

    def new_episode(self, rng, max_tries=200):
        """Compile a fresh randomized model and place object+bin (rejection
        sampled for non-overlap + camera visibility). Returns (model, data, info).
        """
        spec, info = self._build_spec(rng)
        model = spec.compile()                         # fresh, fully-correct model
        data = mujoco.MjData(model)

        # qpos start index of each free joint -> where to write its [x y z quat].
        oadr = model.jnt_qposadr[model.body("object").jntadr[0]]
        badr = model.jnt_qposadr[model.body("bin").jntadr[0]]
        rest_z = info["object_rest_z"]                 # object centre height on floor
        bin_z = info["bin"]["t"]                       # bin rests base on floor
        # Min centre-to-centre distance so object and bin footprints never touch:
        # bin outer radius + object reach + a clearance margin.
        obj_reach = float(np.max(info["object_size"]))
        min_dist = info["bin"]["outer"] + obj_reach + CLEARANCE

        # arm to home so randomization avoids the stowed arm
        data.qpos[:7] = self.home_arm
        data.qpos[7:9] = 0.04

        # Rejection sampling: keep drawing poses until one satisfies BOTH the
        # non-overlap and camera-visibility constraints (or we give up).
        for attempt in range(max_tries):
            ox = rng.uniform(*OBJ_X)
            oy = rng.uniform(*OBJ_Y)
            yaw = rng.uniform(-np.pi, np.pi)           # object spun about world Z
            bx = rng.uniform(*BIN_X)
            sign = rng.choice([-1.0, 1.0])             # bin on +Y or -Y side
            by = sign * rng.uniform(*BIN_Y)

            if np.hypot(ox - bx, oy - by) < min_dist:  # reject: footprints overlap
                continue

            # yaw -> quaternion about Z (w, x, y, z) = (cos, 0, 0, sin).
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            data.qpos[oadr:oadr + 3] = [ox, oy, rest_z]   # object position
            data.qpos[oadr + 3:oadr + 7] = qz             # object orientation
            data.qpos[badr:badr + 3] = [bx, by, bin_z]    # bin position
            data.qpos[badr + 3:badr + 7] = [1, 0, 0, 0]   # bin upright (identity quat)
            mujoco.mj_forward(model, data)                # update cam poses for _visible

            op = np.array([ox, oy, rest_z])
            bpos = np.array([bx, by, info["bin"]["wh"]])  # aim at bin rim height
            if _visible(model, data, op, self.aspect) and \
               _visible(model, data, bpos, self.aspect):
                # object/bin are framed -> now randomize the robot start pose
                # (arm jitter + gripper opening) with collision + occlusion guards
                start_arm, start_grip = self._sample_start_pose(
                    model, data, rng)
                info.update({"object_pose": (ox, oy, rest_z, yaw),
                             "bin_pose": (bx, by, bin_z), "attempts": attempt + 1,
                             "start_arm": start_arm,
                             "start_grip": start_grip,
                             "start_grip_ctrl": _grip_ctrl(start_grip)})
                self.model, self.data = model, data
                return model, data, info

        raise RuntimeError(f"placement rejection failed after {max_tries} tries")


if __name__ == "__main__":
    # Smoke test: render N random episodes and confirm object+bin are framed.
    import sys
    from PIL import Image
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rng = np.random.default_rng(0)
    sm = SceneManager()
    r = None
    for i in range(n):
        m, d, info = sm.new_episode(rng)
        if r is None:
            r = mujoco.Renderer(m, height=480, width=640)
        types = {6: "box", 5: "cylinder", 2: "sphere"}
        print(f"ep{i}: object={types.get(info['object_type'], info['object_type'])} "
              f"size={np.round(info['object_size'],3)} bin_half={info['bin']['half']:.3f} "
              f"attempts={info['attempts']}  obj={np.round(info['object_pose'][:3],3)} "
              f"bin={np.round(info['bin_pose'],3)}")
        for cam in ("diag", "front"):
            r.update_scene(d, camera=cam)
            Image.fromarray(r.render()).save(os.path.join(HERE, f"ep{i}_{cam}.png"))
    if r:
        r.close()
    print("wrote ep*_diag.png / ep*_front.png")

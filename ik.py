"""Inverse kinematics for the Franka Panda, MuJoCo-native (no URDF needed).

The Panda is 7-DOF, redundant for a 6-DOF pose, so it can hold an arbitrary
position AND orientation at once. We use damped least-squares (DLS) on the
site Jacobian from mj_jacSite, with a null-space posture term that pulls the
redundant elbow toward a comfortable rest pose.

  dq = J^T (J J^T + lambda^2 I)^-1 e          # DLS task step
  dq += (I - J^+ J) k (q_rest - q)            # null-space posture

Run:
  python ik.py test     # headless: recover known poses, print accuracy
  python ik.py demo      # animate a reach to a target pose
"""
import os
import sys
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")
SITE = "grip"
ARM_DOF = 7
Q_REST = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class PandaIK:
    """DLS Jacobian IK over the 7 arm joints of the Panda."""

    def __init__(self, model=None, site=SITE):
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(site).id
        rng = self.model.jnt_range[:ARM_DOF].copy()
        self.q_lo, self.q_hi = rng[:, 0], rng[:, 1]
        self._Jp = np.zeros((3, self.model.nv))
        self._Jr = np.zeros((3, self.model.nv))

    def _kinematics(self, q):
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q):
        """Arm joint angles (7) -> (position 3, rotation matrix 3x3)."""
        self._kinematics(q)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].copy().reshape(3, 3))

    def ee_pose(self, q):
        """Arm joint angles (7) -> (position 3, quaternion wxyz)."""
        p, R = self.fk(q)
        return p, mat2quat(R)

    def solve(self, q_init, target_pos, target_quat=None, iters=100,
              damping=0.08, ori_w=1.0, posture_k=0.02, max_dq=None):
        """Solve for arm joint angles driving the grip site to the target.

        target_quat=None -> position-only (orientation free). max_dq caps the
        per-iteration joint step (rad) for smooth teleop; None = uncapped."""
        q = np.array(q_init, float)[:ARM_DOF].copy()
        tgt_R = None
        if target_quat is not None:
            tgt_R = np.zeros(9)
            mujoco.mju_quat2Mat(tgt_R, np.asarray(target_quat, float))
            tgt_R = tgt_R.reshape(3, 3)
        ori_err = np.zeros(3)
        for _ in range(iters):
            self._kinematics(q)
            p = self.data.site_xpos[self.site_id]
            pos_err = target_pos - p
            if tgt_R is None:
                e = pos_err.copy(); nrows = 3
            else:
                # world-frame orientation error (matches mj_jacSite's world Jr):
                # rotation vector of R_target @ R_current^T
                cur_R = self.data.site_xmat[self.site_id].reshape(3, 3)
                q_err = mat2quat(tgt_R @ cur_R.T)
                mujoco.mju_quat2Vel(ori_err, q_err, 1.0)
                e = np.concatenate([pos_err, ori_w * ori_err]); nrows = 6
            mujoco.mj_jacSite(self.model, self.data, self._Jp, self._Jr, self.site_id)
            if tgt_R is None:
                Jc = self._Jp[:, :ARM_DOF]
            else:
                Jc = np.vstack([self._Jp[:, :ARM_DOF], self._Jr[:, :ARM_DOF]])
            JJt = Jc @ Jc.T + (damping ** 2) * np.eye(nrows)
            dq = Jc.T @ np.linalg.solve(JJt, e)
            if posture_k:                                    # null-space posture
                Jpinv = Jc.T @ np.linalg.inv(JJt)
                N = np.eye(ARM_DOF) - Jpinv @ Jc
                dq = dq + N @ (posture_k * (Q_REST - q))
            if max_dq is not None:
                m = np.max(np.abs(dq))
                if m > max_dq:
                    dq *= max_dq / m
            q = np.clip(q + dq, self.q_lo, self.q_hi)
            if np.linalg.norm(e) < 1e-5:
                break
        return q


# ------------------------------------------------------------------ headless test
def run_test():
    ik = PandaIK()
    print(f"site '{SITE}' | arm DOF {ARM_DOF}")
    p0, q0 = ik.ee_pose(Q_REST)
    print("rest pose: pos", np.round(p0, 4), "quat", np.round(q0, 3))
    rng = np.random.default_rng(0)
    pos_errs, ori_errs = [], []
    for _ in range(8):
        # pick a random reachable config, take its FK as the target, then solve
        qr = Q_REST + rng.uniform(-0.5, 0.5, ARM_DOF)
        qr = np.clip(qr, ik.q_lo, ik.q_hi)
        tp, tq = ik.ee_pose(qr)
        sol = ik.solve(Q_REST, tp, tq, iters=200)
        ap, aq = ik.ee_pose(sol)
        pe = np.linalg.norm(ap - tp) * 1000
        oe = np.zeros(3); mujoco.mju_subQuat(oe, tq, aq)
        pos_errs.append(pe); ori_errs.append(np.rad2deg(np.linalg.norm(oe)))
    print(f"full 6D pose recovery (8 targets): pos {np.mean(pos_errs):.3f} mm "
          f"(max {np.max(pos_errs):.3f}) | ori {np.mean(ori_errs):.3f} deg "
          f"(max {np.max(ori_errs):.3f})")


# ------------------------------------------------------------------ animated demo
def run_demo():
    import control
    model, data = control.model, control.data
    ik = PandaIK(model)
    control.reset_home()
    p0, q_quat = ik.ee_pose(data.qpos[:ARM_DOF])
    targets = [p0 + np.array([0.0, 0.2, -0.1]),
               p0 + np.array([0.15, -0.2, 0.05]),
               p0 + np.array([-0.1, 0.0, 0.2]), p0]
    import glfw
    v = control.Viewer("Panda IK demo")
    for tp in targets:
        sol = ik.solve(data.qpos[:ARM_DOF], tp, q_quat, iters=200)
        t0 = time.time()
        start = data.ctrl[:ARM_DOF].copy()
        while (t := time.time() - t0) < 2.0 and v.running():
            if v.held(glfw.KEY_ESCAPE):
                v.close(); return
            a = min(1.0, t / 1.5)
            data.ctrl[:ARM_DOF] = (1 - a) * start + a * sol
            control.clamp(); control.step_physics(); v.render()
    while v.running() and not v.held(glfw.KEY_ESCAPE):
        control.step_physics(); v.render()
    v.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        run_test()
    elif mode == "demo":
        run_demo()
    else:
        print(f"unknown mode '{mode}' (use: test | demo)")
        sys.exit(1)

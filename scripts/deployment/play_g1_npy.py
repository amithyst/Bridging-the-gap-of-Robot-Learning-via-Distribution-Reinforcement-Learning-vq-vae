# scripts/deployment/play_g1_npy.py

"""
Usage:
    python scripts/deployment/play_g1_npy.py --input_file motions/demo_recon_resnet_hybrid_idx0.npy --output_name test_run
"""

import argparse
# === 关键修改：此时不要 import numpy 或 torch ===
# import numpy as np  <-- 移走
# import torch        <-- 移走

from isaaclab.app import AppLauncher

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Replay G1 motion from NPY file.")
parser.add_argument("--input_file", type=str, required=True, help="Path to .npy file (T, 29)")
parser.add_argument("--input_fps", type=int, default=20, help="FPS of the motion data (LAFAN usually 30 or 20)")
parser.add_argument("--output_name", type=str, default="debug", help="Name for log output")
parser.add_argument("--output_fps", type=int, default=50, help="Sim FPS")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app # Launch Omniverse

# ==========================================================
# === 关键修改：仿真器启动之后，再导入 Torch 和 Numpy ===
# ==========================================================
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

import os
import sys
from pathlib import Path
# G1 Configuration
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# =========================================================
# === FIX: 指向项目本地修复版的 URDF ===
# =========================================================
# 获取当前脚本所在路径
current_script_path = Path(__file__).resolve()
# 获取项目根目录 (假设脚本在 scripts/deployment/, 所以向上两级)
project_root = current_script_path.parents[2]

# 构造本地 URDF 的绝对路径
# 注意：这里指向我们在第1步创建的副本
local_urdf_path = project_root / "assets" / "g1_local" / "urdf" / "g1" / "main.urdf"

if not local_urdf_path.exists():
    raise FileNotFoundError(f"本地 URDF 副本未找到，请检查路径: {local_urdf_path}")

print(f"[INFO] Patching G1 URDF path to LOCAL COPY: {local_urdf_path}")
G1_CYLINDER_CFG.spawn.asset_path = str(local_urdf_path)
# =========================================================

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # G1 Robot
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device):
        self.device = device
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        
        # Load Data
        print(f"Loading motion: {motion_file}")
        raw_data = np.load(motion_file) # Expecting (T, 29)
        self.input_frames = raw_data.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        
        # === KEY FIX: Handle Missing Root Motion ===
        # Data shape is (T, 29) -> Joint Positions Only
        # We need to construct Base Pos (3) and Base Rot (4)
        
        # 1. Joint Positions
        self.motion_dof_poss_input = torch.tensor(raw_data, dtype=torch.float32, device=device)
        
        # 2. Fake Root Position (Fixed at standard height)
        # G1 default hip height is approx 0.75m ?? Adjust if needed.
        self.motion_base_poss_input = torch.zeros((self.input_frames, 3), device=device)
        self.motion_base_poss_input[:, 2] = 0.74  # Fix Z height
        
        # 3. Fake Root Rotation (Identity / Facing Forward)
        # [w, x, y, z] = [1, 0, 0, 0]
        self.motion_base_rots_input = torch.zeros((self.input_frames, 4), device=device)
        self.motion_base_rots_input[:, 0] = 1.0 
        
        # Pre-compute interpolation
        self._interpolate_motion()
        self._compute_velocities()
        self.current_idx = 0

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = len(times)
        
        # Linear Interpolation indices
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - idx0
        
        # Interpolate
        # Use simple Linear for joints and pos
        blend_unsqueezed = blend.unsqueeze(1)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1], blend_unsqueezed)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1], blend_unsqueezed)
        
        # Slerp for Rotation (even though it's constant here, good to keep logic)
        self.motion_base_rots = torch.zeros((self.output_frames, 4), device=self.device)
        for i in range(self.output_frames):
            self.motion_base_rots[i] = quat_slerp(self.motion_base_rots_input[idx0[i]], self.motion_base_rots_input[idx1[i]], blend[i])

    def _lerp(self, a, b, t):
        return a * (1 - t) + b * t
        
    def _compute_velocities(self):
        # Finite difference for velocities
        self.motion_base_lin_vels = torch.zeros_like(self.motion_base_poss) # Static root = 0 vel
        self.motion_base_ang_vels = torch.zeros((self.output_frames, 3), device=self.device)
        
        # Compute joint vel
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx = (self.current_idx + 1) % self.output_frames
        return state, (self.current_idx == 0)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # G1 Joint Names (Order matters!)
    # Must match the order in your VQ-VAE training data (29 dims)
    # Check your dataset to confirm this order.
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]
    # 注意：如果你的 VQ-VAE 数据集里关节顺序和这个不一样，这里必须调整列表顺序！
    
    motion_loader = MotionLoader(args_cli.input_file, args_cli.input_fps, args_cli.output_fps, sim.device)
    robot = scene["robot"]

    # ==================================================
    # === 关键修改：必须先 Reset 仿真，才能操作机器人 ===
    # ==================================================
    sim.reset() 
    print("[INFO]: Simulation started & Robot initialized...")

    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    
    while simulation_app.is_running():
        state, reset = motion_loader.get_next_state()
        base_pos, base_rot, base_lin, base_ang, dof_pos, dof_vel = state
        
        # Apply State
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = base_pos
        root_state[:, 3:7] = base_rot # w, x, y, z
        
        joint_pos_target = robot.data.default_joint_pos.clone()
        joint_pos_target[:, robot_joint_indexes] = dof_pos
        
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos_target, robot.data.default_joint_vel.clone())
        
        sim.render()
        scene.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()
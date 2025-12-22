# scripts/deployment/render_viewport.py

import argparse
from isaaclab.app import AppLauncher

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Render G1 motion from Viewport (WYSIWYG).")
parser.add_argument("--input_file", type=str, nargs='+', required=True, help="Path to .npy file(s)")
parser.add_argument("--output_root", type=str, default="plots/rendered_viewport", help="Output directory")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)

# [新增] 限制最大截图数量，-1 表示不限制
parser.add_argument("--max_shots", type=int, default=-1, help="Max screenshots to save (downsample). -1=Save All.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app 

import torch
import numpy as np
import os
from pathlib import Path
import carb

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_slerp

# Viewport Capture
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
import omni.usd
from pxr import Gf, UsdGeom
import imageio  # <--- 新增这行
# G1 Robot
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# Patch URDF
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parents[2]
local_urdf_path = project_root / "assets" / "g1_local" / "urdf" / "g1" / "main.urdf"
if not local_urdf_path.exists():
    pass # 如果没有本地URDF，就用默认的，防止报错中断
else:
    G1_CYLINDER_CFG.spawn.asset_path = str(local_urdf_path)

@configclass
class RenderSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0, 
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file, device):
        self.device = device
        self.dt = 1.0 / 30.0 
        
        print(f"Loading motion: {motion_file}")
        raw_data = np.load(motion_file)
        self.input_frames = raw_data.shape[0]
        
        self.motion_dof_poss = torch.tensor(raw_data, dtype=torch.float32, device=device)
        self.motion_base_poss = torch.zeros((self.input_frames, 3), device=device)
        self.motion_base_poss[:, 2] = 0.74
        self.motion_base_rots = torch.zeros((self.input_frames, 4), device=device)
        self.motion_base_rots[:, 0] = 1.0 
        
        self.total_frames = self.input_frames

    def get_state_at(self, idx):
        safe_idx = max(0, min(idx, self.total_frames - 1))
        return (
            self.motion_base_poss[safe_idx : safe_idx + 1],
            self.motion_base_rots[safe_idx : safe_idx + 1],
            self.motion_dof_poss[safe_idx : safe_idx + 1],
        )

def set_camera_view_native(eye, target):
    stage = omni.usd.get_context().get_stage()
    camera_path = "/World/RenderCam"
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    
    eye_vec = Gf.Vec3d(*eye)
    target_vec = Gf.Vec3d(*target)
    up_vec = Gf.Vec3d(0, 0, 1)
    
    view_mat = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
    world_mat = view_mat.GetInverse()
    
    translate = world_mat.ExtractTranslation()
    quat = world_mat.ExtractRotation().GetQuat()
    
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(translate)
    xform.AddOrientOp().Set(Gf.Quatf(quat))
    
    viewport_api = get_active_viewport()
    if viewport_api:
        viewport_api.camera_path = camera_path

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    scene_cfg = RenderSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    robot = scene["robot"]
    
    # === 关键修复步骤 ===
    print("[INFO] Simulating reset to get internal joint order...")
    sim.reset()
    
    # 1. 自动获取机器人内部关节顺序
    # 既然数据是按内部顺序录制的，我们也按内部顺序写回去
    internal_joint_names = robot.joint_names
    print(f"[DEBUG] Robot Internal Joint Order (Total {len(internal_joint_names)}):")
    print(internal_joint_names[:5], "...") 
    
    # 2. 不再使用手动列表，直接使用 robot.joint_names
    # find_joints 传入它自己，得到的索引自然是 [0, 1, 2, 3...]
    # 这确保了 data[0] -> joint[0], data[1] -> joint[1]
    robot_joint_indexes = robot.find_joints(internal_joint_names, preserve_order=True)[0]
    # ====================

    # === 支持多文件列表 ===
    input_files = args_cli.input_file
    current_file_idx = 0

    def load_task_context(idx):
        f_path = input_files[idx]
        m_loader = MotionLoader(f_path, sim.device)
        t_name = Path(f_path).stem
        o_dir = Path(args_cli.output_root) / t_name
        o_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Processing file {idx+1}/{len(input_files)}: {t_name}")
        print(f"[INFO] Saving frames to: {o_dir}")
        return m_loader, t_name, o_dir

    motion_loader, task_name, output_dir = load_task_context(current_file_idx)
    # ====================

    # === [新增] 计算采样步长 ===
    # 如果指定了 max_shots，计算每隔多少帧截一次图
    capture_step = 1
    if args_cli.max_shots > 0 and motion_loader.total_frames > args_cli.max_shots:
        capture_step = int(motion_loader.total_frames / args_cli.max_shots)
        capture_step = max(1, capture_step) # 保证至少为1
    print(f"[INFO] Capture step: {capture_step} (Total frames: {motion_loader.total_frames})")
    # ==========================

    set_camera_view_native(eye=[3.5, 3.5, 1.4], target=[0.0, 0.0, 0.75])
    
    print("Warming up...")
    for _ in range(30):
        sim.render()
        
    viewport_api = get_active_viewport()

    print(f"Rendering {motion_loader.total_frames} frames...")
    
    frame_idx = 0
    has_captured = False # 标记是否完成了一轮截图

    while simulation_app.is_running():
        # 获取当前帧数据
        state = motion_loader.get_state_at(frame_idx)
        base_pos, base_rot, dof_pos = state
        
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = base_pos
        root_state[:, 3:7] = base_rot
        
        joint_pos_target = robot.data.default_joint_pos.clone()
        joint_pos_target[:, robot_joint_indexes] = dof_pos
        
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos_target, robot.data.default_joint_vel.clone())
        
        scene.update(sim.get_physics_dt())
        sim.render()
        
        # --- [修改] 截图逻辑：增加取模判断 ---
        # 只有当 frame_idx 是 capture_step 的倍数时才截图
        if not has_captured and (frame_idx % capture_step == 0):
            file_path = str(output_dir / f"frame_{frame_idx:04d}.png")
            capture_viewport_to_file(viewport_api, file_path)
            
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}/{motion_loader.total_frames} (Saved)")

        # --- 循环逻辑 ---
        frame_idx += 1
        if frame_idx >= motion_loader.total_frames:
            if not has_captured:
                print(f"[SUCCESS] Rendering {task_name} complete. Generating GIF...")
                
                # === 生成 GIF ===
                png_files = sorted(output_dir.glob("frame_*.png"))
                if png_files:
                    gif_path = output_dir / f"{task_name}.gif"
                    images = [imageio.imread(str(p)) for p in png_files]
                    imageio.mimsave(gif_path, images, fps=30, loop=0)
                    print(f"[SUCCESS] GIF saved to: {gif_path}")
                
                # === 多文件切换逻辑 ===
                # 如果还有下一个文件，切换到下一个
                if current_file_idx < len(input_files) - 1:
                    current_file_idx += 1
                    print(f"[INFO] Switching to next file...")
                    
                    # 重新加载上下文
                    motion_loader, task_name, output_dir = load_task_context(current_file_idx)

                    # === [新增] 切换文件后重新计算步长 ===
                    capture_step = 1
                    if args_cli.max_shots > 0 and motion_loader.total_frames > args_cli.max_shots:
                        capture_step = int(motion_loader.total_frames / args_cli.max_shots)
                        capture_step = max(1, capture_step)
                    # ==================================
                    
                    # 重置状态
                    frame_idx = 0
                    has_captured = False # 新文件允许截图
                    
                    # 跳过本次循环剩余部分，直接开始新文件的第一帧
                    continue 
                else:
                    # 如果是最后一个文件，就停留在这里循环播放
                    print("[INFO] All files processed. Looping preview of the last file...")
                    has_captured = True 
                    frame_idx = 0 
            else:
                # 已经是 has_captured=True (仅针对最后一个文件会在预览模式下运行)
                frame_idx = 0

if __name__ == "__main__":
    main()
    simulation_app.close()
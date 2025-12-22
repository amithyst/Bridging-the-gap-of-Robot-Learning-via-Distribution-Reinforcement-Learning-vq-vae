import argparse
import imageio
import shutil
from isaaclab.app import AppLauncher

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Render .npy motions to MP4 video.")
parser.add_argument("--input_path", type=str, required=True, help="Path to a single .npy file OR a directory containing .npy files")
parser.add_argument("--output_dir", type=str, default="plots/videos", help="Directory to save videos")
parser.add_argument("--fps", type=int, default=30, help="Output video FPS (Lower = Slower motion)")
parser.add_argument("--keep_frames", action="store_true", help="If set, keeps the individual PNG frames")

# App 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app 

# === Imports after App Launch ===
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Isaac Lab & USD
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
import omni.usd
from pxr import Gf, UsdGeom

# Robot Config
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# Patch URDF Path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parents[2]
local_urdf_path = project_root / "assets" / "g1_local" / "urdf" / "g1" / "main.urdf"
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

# === 修复后的相机设置函数 (四元数版) ===
def set_camera_view_quat(eye, target):
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

def run_render(sim, scene, motion_file, output_video_path, fps, keep_frames):
    print(f"Processing: {motion_file}")
    
    # 1. Load Motion
    raw_data = np.load(motion_file)
    if raw_data.ndim == 3: raw_data = raw_data[0] # Handle (1, 64, 29)
    num_frames = raw_data.shape[0]
    
    device = sim.device
    dof_poss = torch.tensor(raw_data, dtype=torch.float32, device=device)
    
    # Fake Base (Fixed height)
    base_poss = torch.zeros((num_frames, 3), device=device)
    base_poss[:, 2] = 0.74
    base_rots = torch.zeros((num_frames, 4), device=device)
    base_rots[:, 0] = 1.0
    
    robot = scene["robot"]
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
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # Temp dir for frames
    temp_dir = Path(output_video_path).parent / "temp_frames"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Reset
    sim.reset()
    # === 调整相机位置: 远一点 ===
    set_camera_view_quat(eye=[3.5, 3.5, 1.4], target=[0.0, 0.0, 0.75])
    
    # Warmup
    for _ in range(10): sim.render()
    
    viewport_api = get_active_viewport()
    frame_files = []

    # Render Loop
    for i in range(num_frames):
        # Set State
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = base_poss[i:i+1]
        root_state[:, 3:7] = base_rots[i:i+1]
        
        joint_target = robot.data.default_joint_pos.clone()
        joint_target[:, robot_joint_indexes] = dof_poss[i:i+1]
        
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_target, robot.data.default_joint_vel.clone())
        
        scene.update(sim.get_physics_dt())
        sim.render()
        
        # Capture
        file_path = temp_dir / f"frame_{i:04d}.png"
        capture_viewport_to_file(viewport_api, str(file_path))
        frame_files.append(str(file_path))
    
    # Create Video using imageio
    print(f"Stitching {len(frame_files)} frames to MP4...")
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=None)
    
    for filename in sorted(frame_files):
        image = imageio.imread(filename)
        writer.append_data(image)
    writer.close()
    
    if not keep_frames:
        shutil.rmtree(temp_dir)
    
    print(f"[Done] Video saved: {output_video_path}")

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    scene_cfg = RenderSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    output_root = Path(args_cli.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args_cli.input_path)
    files_to_process = []
    
    if input_path.is_dir():
        files_to_process = sorted(list(input_path.glob("*.npy")))
    else:
        files_to_process = [input_path]
        
    print(f"Found {len(files_to_process)} files to render.")

    for f in tqdm(files_to_process):
        video_name = f.stem + ".mp4"
        out_path = output_root / video_name
        run_render(sim, scene, str(f), str(out_path), args_cli.fps, args_cli.keep_frames)

if __name__ == "__main__":
    main()
    simulation_app.close()
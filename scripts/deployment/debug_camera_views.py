# scripts/deployment/debug_camera_views.py

"""
Usage:
    python scripts/deployment/debug_camera_views.py --input_file motions/demo_recon_resnet_hybrid_idx0.npy
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug G1 camera views.")
parser.add_argument("--input_file", type=str, required=True, help="Path to .npy file")
parser.add_argument("--output_root", type=str, default="plots/camera_debug", help="Root directory for debug images")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ==================================================
# === 关键修复：强制开启相机渲染 ===
# ==================================================
args_cli.enable_cameras = True 
# ==================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app 

# Imports
import torch
import numpy as np
from PIL import Image
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_matrix

# === 新增以下 3 行 ===
from pxr import Usd, UsdGeom, Gf
import omni.usd
# ===================

# G1 Robot
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# Patch URDF
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parents[2]
local_urdf_path = project_root / "assets" / "g1_local" / "urdf" / "g1" / "main.urdf"
if not local_urdf_path.exists():
    raise FileNotFoundError(f"Local URDF not found: {local_urdf_path}")
G1_CYLINDER_CFG.spawn.asset_path = str(local_urdf_path)

@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0, 
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 相机传感器
    tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/RunCamera",
        update_period=0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

def set_camera_pose(camera, eye_pos, target_pos, device):
    """
    [最终修复版] 使用 UsdGeom.Xformable 底层接口。
    强制清空原有的变换栈 (ClearXformOpOrder)，然后重建 Translate 和 Rotate。
    这能解决 'incompatible xformable' 报错。
    """
    stage = omni.usd.get_context().get_stage()
    prim_path = "/World/RunCamera"
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        print(f"[ERROR] Camera prim not found at {prim_path}")
        return

    # 1. 计算 LookAt 旋转 (USD 相机默认看向 -Z)
    eye_vec = Gf.Vec3d(float(eye_pos[0]), float(eye_pos[1]), float(eye_pos[2]))
    target_vec = Gf.Vec3d(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
    up_vec = Gf.Vec3d(0, 0, 1)

    # 计算 View Matrix 并取逆得到 World Matrix
    view_mat = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
    model_mat = view_mat.GetInverse()

    # 提取平移和旋转
    translation = model_mat.ExtractTranslation()
    # 提取欧拉角 (Vec3d)
    rotation = model_mat.ExtractRotation().Decompose(Gf.Vec3d(1,0,0), Gf.Vec3d(0,1,0), Gf.Vec3d(0,0,1))
    
    # 2. === 关键修改：使用 Xformable 底层接口 ===
    xformable = UsdGeom.Xformable(prim)
    
    # A. 清空现有的所有变换 (解决 incompatible 报错的核心)
    xformable.ClearXformOpOrder()
    
    # B. 重新添加 Translate 操作
    # AddTranslateOp(precision=precisionDouble)
    xform_op_t = xformable.AddTranslateOp()
    xform_op_t.Set(translation)
    
    # C. 重新添加 RotateXYZ 操作
    # 注意：ExtractRotation 返回的顺序通常适配 XYZ，如果不对可以尝试 AddRotateYXZOp 等
    xform_op_r = xformable.AddRotateXYZOp()
    xform_op_r.Set(Gf.Vec3d(rotation[0], rotation[1], rotation[2]))

    # (可选) 如果 Isaac Sim 还是没刷新，强制标记为已更改
    # prim.GetAttribute("xformOp:translate").Set(translation)
    
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    scene_cfg = DebugSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    robot = scene["robot"]
    camera = scene["tiled_camera"]

    # =========================================================
    # === 关键修改：必须先 Reset 初始化物理引擎，才能找关节 ===
    # =========================================================
    print("[INFO] Initializing simulation...")
    sim.reset()
    # =========================================================
    
    # 关节列表
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
    
    # 加载数据 (只读第一帧)
    raw_data = np.load(args_cli.input_file)
    dof_pos_0 = torch.tensor(raw_data[0], dtype=torch.float32, device=sim.device)
    
    
    # 设置机器人为第0帧状态 (保持不动)
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 0.74 # Base Height
    root_state[:, 3] = 1.0  # Rotation (w=1)
    joint_pos_target = robot.data.default_joint_pos.clone()
    joint_pos_target[:, robot_joint_indexes] = dof_pos_0
    
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos_target, robot.data.default_joint_vel.clone())
    scene.update(sim.get_physics_dt())
    
    # 输出目录
    output_dir = Path(args_cli.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 正在生成调试视图，请查看: {output_dir}")

    # ==========================================
    # 定义要测试的视角列表 (Eye, Target, Name)
    # ==========================================
    # G1 大概在 (0,0,0) 位置，身高约 1.3m
    # 目标点 target 设为 (0, 0, 0.75) 大概是腰部位置
    target_center = [0.0, 0.0, 0.75]
    
    test_views = [
        # 1. 正前方平视
        {"eye": [2.5, 0.0, 0.9], "target": target_center, "name": "front_flat"},
        # 2. 正前方稍微俯视 (经典)
        {"eye": [3.0, 0.0, 1.5], "target": target_center, "name": "front_high"},
        # 3. 侧面 45度
        {"eye": [2.0, 2.0, 1.2], "target": target_center, "name": "side_45"},
        # 4. 正侧面
        {"eye": [0.0, 2.5, 1.0], "target": target_center, "name": "side_flat"},
        # 5. 远景 (检查是不是太近了)
        {"eye": [5.0, 0.0, 2.0], "target": target_center, "name": "front_far"},
        # 6. 后方视角 (万一机器人背对了)
        {"eye": [-2.5, 0.0, 1.2], "target": target_center, "name": "back"},
    ]

    # 预热渲染
    for _ in range(10):
        sim.render()

    # 循环拍摄
    for view in test_views:
        eye = view["eye"]
        target = view["target"]
        name = view["name"]
        
        # 1. 设置相机
        print(f"[Move] Setting camera to {name}...")
        set_camera_pose(camera, eye, target, sim.device)
        
        # 2. 渲染几帧让光线稳定 (Isaac Sim 有时需要几帧来更新 Viewport)
        for _ in range(5):
            robot.write_root_state_to_sim(root_state) # 保持机器人不动
            robot.write_joint_state_to_sim(joint_pos_target, robot.data.default_joint_vel.clone())
            scene.update(sim.get_physics_dt())
            sim.render()
            
        # 3. 拍照
        rgb_tensor = camera.data.output["rgb"][0]
        img_np = rgb_tensor.cpu().numpy()
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
            
        img = Image.fromarray(img_np.astype('uint8'))
        
        # 文件名包含坐标，方便你复制
        filename = f"{name}_eye_{eye[0]}_{eye[1]}_{eye[2]}.png"
        img.save(output_dir / filename)
        print(f"[Saved] {filename}")

    print("[DONE] 所有视角调试图已生成。")

if __name__ == "__main__":
    main()
    simulation_app.close()
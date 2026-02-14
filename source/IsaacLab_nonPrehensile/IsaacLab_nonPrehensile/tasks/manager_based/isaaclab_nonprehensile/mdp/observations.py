# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Lightweight profiling utilities for observation functions
import time
from functools import wraps

def _ensure_obs_timers(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_obs_timers"):
        env._obs_timers = {}
    return env._obs_timers

def profile_obs(fn):
    @wraps(fn)
    def wrapper(env, *args, **kwargs):
        timers = _ensure_obs_timers(env)
        name = fn.__name__
        t0 = time.perf_counter()
        result = fn(env, *args, **kwargs)
        dt = time.perf_counter() - t0
        entry = timers.get(name)
        if entry is None:
            timers[name] = {"time": dt, "count": 1}
        else:
            entry["time"] += dt
            entry["count"] += 1
        return result
    return wrapper

def print_obs_timers(env: "ManagerBasedRLEnv") -> None:
    timers = getattr(env, "_obs_timers", {})
    if not timers:
        print("[obs timers] no data collected yet")
        return
    print("[obs timers] summary:")
    for name, entry in timers.items():
        total = entry["time"]
        count = entry["count"]
        avg = total / count if count > 0 else 0.0
        print(f"  {name}: total={total:.6f}s count={count} avg={avg:.6f}s")

# Debug: print observation stats every N calls
DEBUG_OBS_EVERY = 10000000

_HAND_GOAL_MEAN = torch.tensor([0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # z mean = 0.15
_HAND_GOAL_STD = torch.tensor([0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

def _dbg(env: "ManagerBasedRLEnv", name: str, tensor: torch.Tensor) -> torch.Tensor:
    cnt = getattr(env, "_dbg_obs_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        t = tensor
        if t.dim() >= 2:
            mins = t.min(dim=0).values
            maxs = t.max(dim=0).values
            print(f"[obs] {name}: min={mins.tolist()} max={maxs.tolist()} shape={tuple(t.shape)}")
        else:
            print(f"[obs] {name}: min={t.min().item():.3f}, max={t.max().item():.3f}, shape={tuple(t.shape)}")
    env._dbg_obs_cnt = cnt + 1
    return tensor


def _dbg_cloud(env: "ManagerBasedRLEnv", name: str, cloud_env: torch.Tensor) -> None:
    cnt = getattr(env, "_dbg_cloud_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        # cloud_env shape: (num_envs, num_points, 3)
        x = cloud_env[..., 0]
        y = cloud_env[..., 1]
        z = cloud_env[..., 2]
        print(
            f"[obs] {name}: x[min={x.min().item():.3f}, max={x.max().item():.3f}] "
            f"y[min={y.min().item():.3f}, max={y.max().item():.3f}] "
            f"z[min={z.min().item():.3f}, max={z.max().item():.3f}], shape={tuple(cloud_env.shape)}"
        )
    env._dbg_cloud_cnt = cnt + 1


@profile_obs
def hand_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hand state observation (9D: position[3] + rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9) containing [x,y,z, r11,r12,r13, r21,r22,r23]
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get EE position and orientation in world coordinates
    ee_pos_l_w = ee_frame.data.target_pos_w[..., 1, :]  # (num_envs, 3)
    ee_pos_r_w = ee_frame.data.target_pos_w[..., 2, :]  # (num_envs, 3)
    ee_pos_w = (ee_pos_l_w + ee_pos_r_w) / 2
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    # Convert to environment coordinates
    ee_pos_env = ee_pos_w - env.scene.env_origins
    
    # Rotation as 6D
    rot_matrix = matrix_from_quat(ee_quat_w)  # (num_envs, 3, 3)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    hand_state_9d = torch.cat([ee_pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters (similar to corn config)
        device = hand_state_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        hand_state_9d = (hand_state_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "hand_state", hand_state_9d)


@profile_obs
def robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot state observation (14D: joint_positions[7] + joint_velocities[7]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 14)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get joint positions and velocities
    joint_pos = asset.data.joint_pos[:, :7]
    joint_vel = asset.data.joint_vel[:, :7]
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Normalize joint positions using soft limits around default pos -> [-1,1]
        default_pos = asset.data.default_joint_pos[:, :7]
        soft_limits = asset.data.soft_joint_pos_limits[:, :7, :]
        mins = soft_limits[..., 0]
        maxs = soft_limits[..., 1]
        centers = default_pos
        half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
        pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

        # Normalize joint velocities to [0,1] using soft velocity limits
        vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, :7], min=1e-6)
        vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
        vel_norm = (vel_norm + 1.0) * 0.5
        
        return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))
    else:
        # Return raw joint states without normalization
        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))


@profile_obs
def abs_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
) -> torch.Tensor:
    """Absolute pose goal observation (9D: target position[3] + target rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9)
    """
    from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
    
    target_goal = env.command_manager.get_command(command_name)
    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    # Command now directly contains quaternions, convert to rotation matrix
    rot_matrix = matrix_from_quat(target_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    goal_9d = torch.cat([target_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use goal-specific normalization parameters (similar to corn config)
        device = goal_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        goal_9d = (goal_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "abs_goal", goal_9d)


def rel_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Relative pose goal observation (9D: goal relative to current object pose)."""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate, matrix_from_quat

    target_goal = env.command_manager.get_command(command_name)  # (num_envs, 7)
    object_pose_7d = object_pose_in_env_frame(env, object_cfg)
    obj_pos_env = object_pose_7d[:, :3]
    obj_quat_w = object_pose_7d[:, 3:7]

    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    rel_pos = target_pos - obj_pos_env
    current_quat_inv = quat_conjugate(obj_quat_w)
    rel_quat = quat_mul(target_quat, current_quat_inv)
    rot_matrix = matrix_from_quat(rel_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine relative position and rotation
    rel_pose_9d = torch.cat([rel_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for relative pose goal
        device = rel_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        rel_pose_9d = (rel_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return rel_pose_9d


def object_pose_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose of the object in the environment coordinate frame.

    Returns:
        torch.Tensor: Shape (num_envs, 7) containing [x, y, z, qw, qx, qy, qz] in environment coordinates
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_quat_w = object.data.root_quat_w       # (num_envs, 4)
    object_pos_env = object_pos_w - env.scene.env_origins  # (num_envs, 3)
    pose_7d = torch.cat([object_pos_env, object_quat_w], dim=1)  # (num_envs, 7)

    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for object pose
        device = pose_7d.device
        # For 7D pose: position [x,y,z] + quaternion [qw,qx,qy,qz]
        # Use position normalization from hand_goal params
        pos_mean = _HAND_GOAL_MEAN[:3].to(device).view(1, 3)  # [x, y, z] mean
        pos_std = _HAND_GOAL_STD[:3].to(device).view(1, 3)    # [x, y, z] std
        # For quaternion, use simple normalization (quaternions are already normalized)
        quat_mean = torch.zeros(4, device=device).view(1, 4)  # [qw, qx, qy, qz] mean
        quat_std = torch.ones(4, device=device).view(1, 4)    # [qw, qx, qy, qz] std
        
        # Normalize position and quaternion separately
        pos_norm = (pose_7d[:, :3] - pos_mean) / torch.clamp(pos_std, min=1e-6)
        quat_norm = (pose_7d[:, 3:7] - quat_mean) / torch.clamp(quat_std, min=1e-6)
        
        pose_7d = torch.cat([pos_norm, quat_norm], dim=1)
        
    return pose_7d


def object_pose_9d_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The object's current pose in the environment frame as 9D [x,y,z, r11,r12,r13, r21,r22,r23]."""
    pose_7d = object_pose_in_env_frame(env, object_cfg)
    pos_env = pose_7d[:, :3]
    quat_wxyz = pose_7d[:, 3:7]

    # Convert to rotation matrix
    rot_matrix = matrix_from_quat(quat_wxyz)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    object_pose_9d = torch.cat([pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand/goal-specific normalization parameters for object pose too
        device = object_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        object_pose_9d = (object_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return object_pose_9d
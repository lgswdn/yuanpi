# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for non-prehensile manipulation environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_initial_object_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset cube position with random XY position and yaw-only rotation.

    Simplified version for cube pushing - only randomizes position and yaw rotation.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        asset_cfg: Asset configuration.
    """
    # Get the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get curriculum ranges from command manager
    stable_pose_term = env.command_manager.get_term("target_object_pose")
    xy_range = stable_pose_term.initial_position_range

    # Define base position (center of table in environment coordinates)
    base_x = 0.5  # center of table x-coordinate
    base_y = 0.0  # center of table y-coordinate
    base_z = 0.025  # half of cube height (5cm cube, so 2.5cm above table)

    # Sample random positions within curriculum ranges
    num_resets = len(env_ids)
    # Create poses: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    poses = torch.zeros((num_resets, 13), device=env.device)

    for i, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())

        # Random XY position around base
        dx = sample_uniform(-xy_range, xy_range, (1,), device=env.device).squeeze(0)
        dy = sample_uniform(-2*xy_range, 2*xy_range, (1,), device=env.device).squeeze(0)
        x_env = torch.as_tensor(base_x, device=env.device) + dx
        y_env = torch.as_tensor(base_y, device=env.device) + dy

        # Convert to world coordinates
        pos_x = x_env + env.scene.env_origins[env_id_int, 0]
        pos_y = y_env + env.scene.env_origins[env_id_int, 1]
        pos_z = torch.as_tensor(base_z, device=env.device)

        # Random yaw rotation only (rotation around Z-axis)
        yaw = (torch.rand(1, device=env.device) * (2 * torch.pi) - torch.pi).squeeze(0)

        # Create quaternion for yaw rotation: [w, x, y, z]
        qw = torch.cos(yaw * 0.5)
        qx = torch.tensor(0.0, device=env.device)
        qy = torch.tensor(0.0, device=env.device)
        qz = torch.sin(yaw * 0.5)

        # Fill pose row
        poses[i, 0] = pos_x
        poses[i, 1] = pos_y
        poses[i, 2] = pos_z
        poses[i, 3] = qw
        poses[i, 4] = qx
        poses[i, 5] = qy
        poses[i, 6] = qz
        # velocities already zeros

    # Apply the new poses
    asset.write_root_state_to_sim(poses, env_ids)

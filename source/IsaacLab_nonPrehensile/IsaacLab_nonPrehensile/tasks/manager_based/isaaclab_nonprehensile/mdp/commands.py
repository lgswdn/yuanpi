# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Lightweight profiling utilities for command functions
import time
from functools import wraps

def _ensure_cmd_timers(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_cmd_timers"):
        env._cmd_timers = {}
    return env._cmd_timers

def profile_cmd(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        timers = _ensure_cmd_timers(self._env)
        name = fn.__name__
        t0 = time.perf_counter()
        result = fn(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        entry = timers.get(name)
        if entry is None:
            timers[name] = {"time": dt, "count": 1}
        else:
            entry["time"] += dt
            entry["count"] += 1
        return result
    return wrapper

def print_cmd_timers(env: "ManagerBasedRLEnv") -> None:
    timers = getattr(env, "_cmd_timers", {})
    if not timers:
        print("[cmd timers] no data collected yet")
        return
    print("[cmd timers] summary:")
    for name, entry in timers.items():
        total = entry["time"]
        count = entry["count"]
        avg = total / count if count > 0 else 0.0
        print(f"  {name}: total={total:.6f}s count={count} avg={avg:.6f}s")


class StablePoseCommand(CommandTerm):
    """Command generator for stable object poses using trimesh."""

    cfg: "StablePoseCommandCfg"

    def __init__(self, cfg: "StablePoseCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # create command buffer (7D: position + quaternion)
        self._command = torch.zeros(self.num_envs, 7, device=self.device)

        # Curriculum parameters - can be modified dynamically
        self.xy_offset_range = self.cfg.xy_offset_range
        self.initial_position_range = self.cfg.initial_position_range

        # metrics
        self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rot_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        
        # create visualization markers for target poses if debug visualization is enabled
        if self.cfg.debug_vis:
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/goal_pose"
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Make frames smaller
            self._target_visualizer = VisualizationMarkers(marker_cfg)
        else:
            self._target_visualizer = None

    @profile_cmd
    def _resample_command(self, env_ids: torch.Tensor):
        """Resample commands for given environment indices.

        Simplified for cube pushing - only generates random XY positions and yaw rotations.
        """
        # Define base position (center of table in environment coordinates)
        base_x = 0.5
        base_y = 0.0
        base_z = 0.025  # half of cube height (5cm cube)

        # Preallocate output tensor on device
        num_resets = len(env_ids)
        out = torch.zeros((num_resets, 7), dtype=torch.float32, device=self.device)

        # Generate random positions and yaw rotations for all environments
        r = self.xy_offset_range
        sign_x = torch.where(torch.rand(num_resets, device=self.device) < 0.5, -1.0, 1.0)
        sign_y = torch.where(torch.rand(num_resets, device=self.device) < 0.5, -1.0, 1.0)
        mag_x = (0.5 + torch.rand(num_resets, device=self.device) * 0.5) * r
        mag_y = (0.5 + torch.rand(num_resets, device=self.device) * 0.5) * r
        x_env = base_x + sign_x * mag_x
        y_env = base_y + 2.0 * (sign_y * mag_y)

        # Random yaw rotations only (rotation around Z-axis)
        yaw = (torch.rand(num_resets, device=self.device) * (2 * torch.pi) - torch.pi)

        # Create quaternions for yaw rotations: [w, x, y, z]
        qw = torch.cos(yaw * 0.5)
        qx = torch.zeros(num_resets, device=self.device)
        qy = torch.zeros(num_resets, device=self.device)
        qz = torch.sin(yaw * 0.5)

        # Build pose_7d: [x, y, z, qw, qx, qy, qz]
        out[:, 0] = x_env
        out[:, 1] = y_env
        out[:, 2] = base_z
        out[:, 3] = qw
        out[:, 4] = qx
        out[:, 5] = qy
        out[:, 6] = qz

        # Write results to command buffer
        for i, env_id in enumerate(env_ids):
            env_id_int = int(env_id.item())
            self.command[env_id_int] = out[i]

        # Update visualization after resampling commands
        if self.cfg.debug_vis and self._target_visualizer is not None:
            self._update_visualization()
                        
    def _update_command(self):
        """Update the command. No-op for this implementation."""
        pass

    def _update_visualization(self):
        """Update the visualization of target poses."""
        if self.cfg.debug_vis and self._target_visualizer is not None:
            local_positions = self.command[:, :3].clone()
            quaternions = self.command[:, 3:7].clone()
            
            # Convert local positions to world positions by adding environment origins
            world_positions = local_positions + self._env.scene.env_origins
            
            # visualize target pose frames using world positions
            self._target_visualizer.visualize(translations=world_positions, orientations=quaternions)

    def _update_metrics(self):
        """Update the metrics."""
        # compute distance to goal (simplified)
        object_asset: RigidObject = self._env.scene["object"]
        # Get object position in local coordinates (relative to environment origin)
        object_pos_world = object_asset.data.root_pos_w[:, :3]
        object_pos_local = object_pos_world - self._env.scene.env_origins
        
        # Target position is already in local coordinates
        target_pos_local = self.command[:, :3]
        
        self.metrics["distance_to_goal"] = torch.norm(object_pos_local - target_pos_local, dim=-1)

        # --- Rotation matrix distance metric ---
        # Get current and target orientation
        object_quat_w = object_asset.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
        # Target quaternion (in command)
        target_quat = self.command[:, 3:7]  # [w, x, y, z]

        # Compute quaternion angular distance (in radians)
        dot_product = torch.sum(object_quat_w * target_quat, dim=1)
        dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
        quat_angle = 2 * torch.acos(dot_product)
        self.metrics["rot_to_goal"] = quat_angle

    @property
    def command(self) -> torch.Tensor:
        """The generated command."""
        return self._command

    @command.setter
    def command(self, value: torch.Tensor):
        """Set the command."""
        self._command = value.clone()


@configclass
class StablePoseCommandCfg(CommandTermCfg):
    """Configuration for stable pose command generator."""

    class_type: type = StablePoseCommand
    
    # Curriculum-controllable parameters
    xy_offset_range: float = 0.2
    """Random XY offset range for goal position in meters. Default: ±20cm"""
    
    initial_position_range: float = 0.15
    """Random XY offset range for initial object position in meters. Default: ±15cm"""
    
    # Visualization
    debug_vis: bool = False
    """Whether to visualize the target poses."""

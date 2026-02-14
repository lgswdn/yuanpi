# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.terrains import TerrainImporterCfg
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG


default_joint_pos = FRANKA_PANDA_HIGH_PD_CFG.init_state.joint_pos.copy()
# User-defined joint workspace for Franka arm (7 DOF)
JOINT_BOX_MIN_BASE = [-0.3, -0.4636, -0.2, -2.7432, -0.3335, 1.5269, -1.5707963267948966]
JOINT_BOX_MAX_BASE = [0.3, 0.5432, 0.2, -1.5237, 0.3335, 2.5744, 1.5707963267948966]
JOINT_BOX_SHIFT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT_BOX_MIN = [mn + d for mn, d in zip(JOINT_BOX_MIN_BASE, JOINT_BOX_SHIFT)]
JOINT_BOX_MAX = [mx + d for mx, d in zip(JOINT_BOX_MAX_BASE, JOINT_BOX_SHIFT)]
# Choose initial pose as midpoint within the box range
_joint_init_mid = [(mn + mx) / 2.0 for mn, mx in zip(JOINT_BOX_MIN, JOINT_BOX_MAX)]
custom_joint_init = {
    "panda_joint1": _joint_init_mid[0],
    "panda_joint2": _joint_init_mid[1],
    "panda_joint3": _joint_init_mid[2],
    "panda_joint4": _joint_init_mid[3],
    "panda_joint5": _joint_init_mid[4],
    "panda_joint6": _joint_init_mid[5],
    "panda_joint7": _joint_init_mid[6],
    "panda_finger_joint.*": 0.0,
}


@configclass
class NonPrehensileSceneCfg(InteractiveSceneCfg):
    """Configuration for a non-prehensile scene."""
    
    # Disable physics replication to avoid conflicts with MultiAssetSpawnerCfg
    replicate_physics: bool = False
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # optional: "plane", "usd", "generator"
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        debug_vis=False,
    )
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(0.8, 0.6, 1.0),
    #     ),
    # )
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(2.0, 3.0, 1.0),
    #     ),
    # )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=custom_joint_init
        ),
        spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
            activate_contact_sensors=True
        )
    )
    # end-effector sensor: will be populated by agent env cfg
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    # Simple cube object for pushing task
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm cube
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),  # Fixed mass: 200g
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),  # Red cube
        ),
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    target_object_pose = mdp.StablePoseCommandCfg(
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,  # Visualize target pose
        xy_offset_range=0.2,
        initial_position_range=0.2,
    )

@configclass
class RelativeJointPositionActionsCfg:
    """Relative (delta) joint position action specifications for the MDP."""
    # Relative joint position control: q_target = q_current + scaled_action
    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.1,
        use_zero_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Cube State (7D: position[3] + quaternion[4] in environment frame)
        # For a simple cube, we just need its position and yaw rotation
        cube_state = ObsTerm(
            func=mdp.object_pose_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        # Hand State (9D: hand position[3] + rotation_matrix[6])
        hand_state = ObsTerm(
            func=mdp.hand_state, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        # Robot State (14D: joint_positions[7] + joint_velocities[7])
        robot_state = ObsTerm(
            func=mdp.robot_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        # Previous Action (Variable D: depends on action type)
        previous_action = ObsTerm(func=mdp.last_action)

        # Relative Pose Goal (9D: goal relative to current object pose)
        rel_goal = ObsTerm(
            func=mdp.rel_pose_goal, params={"command_name": "target_object_pose"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Only keep position randomization
    reset_object_position = EventTerm(
        func=mdp.reset_initial_object_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    task_success = RewTerm(
        func=mdp.task_success_reward,
        params={
            "command_name": "target_object_pose", 
            "threshold": 0.05, 
            "rotation_threshold": 0.1, 
            "planar": False,
            "base_reward": 1.0,  # Base reward for success
            "time_bonus_factor": 0.5,  # Bonus factor for early completion
        },
        weight=2000.0
    )

    contact_reward = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={
            "std": 0.1,
        },
        weight=1.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": 0.5,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": 0.05,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=5.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": 0.2,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": 0.05,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=16.0,
    )
    
    # Energy penalty: c_energy = k_e * Σ(τ_i * q̇_i)
    # energy_penalty = RewTerm(
    #     func=mdp.joint_power_penalty,
    #     params={"k_e": 0.0001},  # scaling coefficient
    #     weight=-1.0,  # negative weight for penalty
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "target_object_pose", "threshold": 0.05, "rotation_threshold": 0.1, "planar": False},
    )
    object_dropped = DoneTerm(
        func=mdp.object_dropped_off_table,
        params={"minimum_height": -0.15}  # 15cm below table surface
    )

@configclass
class NonPrehensileEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: NonPrehensileSceneCfg = NonPrehensileSceneCfg(num_envs=64, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: RelativeJointPositionActionsCfg = RelativeJointPositionActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Observation normalization
    normalize_observations: bool = True  # Whether to normalize observations to [-1,1] range, except hand_state and pointcloud 
    # Visualization settings
    visualize_current_object_pose: bool = True  # Enable current object pose visualization
    visualize_object_pointcloud: bool = False  # Enable object point cloud visualization for debug in first env

    # Performance settings
    use_torch_compile: bool = True  # Enable torch.compile on hot paths

    # Whether to enforce (apply) the robot soft joint limits configured in this env
    # Set to False to skip updating soft joint limits at environment creation time.
    enforce_joint_limits: bool = False

    # Disable observation noise across all policy observation terms during env creation
    disable_obs_noise: bool = False

    def __post_init__(self) -> None:
        # Optionally disable observation noise for evaluation or ablations
        if self.disable_obs_noise:
            obs_cfg = self.observations
            policy_cfg = getattr(obs_cfg, "policy", None)
            if policy_cfg is not None:
                for attr_name in dir(policy_cfg):
                    if attr_name.startswith("_"):
                        continue
                    term = getattr(policy_cfg, attr_name, None)
                    if term is None:
                        continue
                    noise = getattr(term, "noise", None)
                    if noise is None:
                        continue
                    if hasattr(noise, "mean"):
                        noise.mean = 0.0
                    if hasattr(noise, "std"):
                        noise.std = 0.0
        
        # General settings
        self.decimation = 8
        self.episode_length_s = 10  # Reduced from 30s to 10s for faster cube pushing
        
        # Viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        
        # Simulation settings - match reference config dt
        self.sim.dt = 1 / 80
        self.sim.render_interval = self.decimation
        
        # Physics settings - match reference config
        self.sim.physx.solver_position_iteration_count = 8  # pos_iter=8
        self.sim.physx.solver_velocity_iteration_count = 1  # vel_iter=1


class NonPrehensileEnv(ManagerBasedRLEnv):
    """Custom environment wrapper for non-prehensile manipulation.
    
    This class extends ManagerBasedRLEnv and relies on IsaacLab's built-in
    action tracking via action_manager.action for previous action observations.
    """
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Initialize the base environment
        super().__init__(cfg, render_mode, **kwargs)
        # Override Franka arm soft joint limits with user-defined box and clamp current state
        robot = self.scene["robot"]
        entity = SceneEntityCfg("robot", joint_names=["panda_joint.*"])  # 7 arm joints
        entity.resolve(self.scene)
        joint_ids = entity.joint_ids
        mins = torch.tensor(JOINT_BOX_MIN, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        maxs = torch.tensor(JOINT_BOX_MAX, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        # update soft limits in-place
        if self.cfg.enforce_joint_limits:
            limits = robot.data.soft_joint_pos_limits
            limits[:, joint_ids, 0] = mins
            limits[:, joint_ids, 1] = maxs
            robot.data.soft_joint_pos_limits[:] = limits

        # Initialize success tracking buffers
        self.episode_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.total_episodes = 0
        self.total_successes = 0

        # Sliding window for recent success rate (last 100 episodes)
        self.recent_success_window = deque(maxlen=100)
        self.recent_success_rate = 0.0

        # Global step counter for periodic debug prints
        self._global_step = 0
    
    def step(self, action):
        """Override step to track success rates."""
        # Call parent step method
        obs, reward, terminated, truncated, info = super().step(action)

        # Increment global step and periodically print timers
        # self._global_step += 1
        # if self._global_step % 1 == 0:
        #     from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import print_obs_timers
        #     from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.commands import print_cmd_timers
        #     print_obs_timers(self)
        #     print_cmd_timers(self)

        success_mask = self.termination_manager.get_term("reached")
        # Update episode success buffer
        self.episode_success_buf = self.episode_success_buf | success_mask
        
        # Check for episode endings (terminated or truncated)
        episode_ended = terminated | truncated
        
        # Update success statistics when episodes end
        if torch.any(episode_ended):
            ended_env_ids = torch.where(episode_ended)[0]
            for env_id in ended_env_ids:
                self.total_episodes += 1
                episode_success = self.episode_success_buf[env_id].item()
                if episode_success:
                    self.total_successes += 1
                
                # Add to sliding window for recent success rate
                self.recent_success_window.append(episode_success)
            
            # Store success status before reset for external access
            self._episode_success_before_reset = self.episode_success_buf.clone()
            
            # Reset success buffer for ended episodes
            self.episode_success_buf[episode_ended] = False
            
            # Calculate and log success rate
            if self.total_episodes > 0:
                success_rate = self.total_successes / self.total_episodes
                # Add to extras["log"] for tensorboard logging
                if "log" not in self.extras:
                    self.extras["log"] = dict()
                self.extras["log"]["success_rate"] = success_rate
                self.extras["log"]["total_episodes"] = self.total_episodes
                self.extras["log"]["total_successes"] = self.total_successes
                
                # Calculate recent success rate (sliding window of last 100 episodes)
                if len(self.recent_success_window) > 0:
                    self.recent_success_rate = sum(self.recent_success_window) / len(self.recent_success_window)
                    self.extras["log"]["recent_success_rate"] = self.recent_success_rate

        return obs, reward, terminated, truncated, info
    

    def post_reset(self):
        self.sim.physx.bounce_threshold_velocity = 0.05
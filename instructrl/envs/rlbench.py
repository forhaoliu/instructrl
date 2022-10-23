import random
from functools import partial
from typing import Any, Optional, Tuple, Union

import gym
import numpy as np
from ml_collections import ConfigDict
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Dummy, VisionSensor
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    JointPosition,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete, GripperJointPosition
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from yarr.agents.agent import ActResult, VideoSummary

from instructrl.envs.custom import KeypointEnvironment


class RLBenchArmGripper:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.arm_gripper_mode = "gripper_pose"
        config.image_key = "front_rgb, left_shoulder_rgb, right_shoulder_rgb, wrist_rgb"
        config.state_key = ""
        config.absolute_mode = False
        config.episode_length = 100
        config.variation = 0
        config.record_video = False
        config.record_every = 5

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(
        self,
        game_name: str,
        update: ConfigDict,
    ):
        self.config = self.get_default_config(update)

        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._i = 0

        img_size = (256, 256)

        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.right_shoulder_camera.image_size = img_size
        obs_config.left_shoulder_camera.image_size = img_size
        obs_config.overhead_camera.image_size = img_size
        obs_config.wrist_camera.image_size = img_size
        obs_config.front_camera.image_size = img_size

        # Store depth as 0 - 1
        obs_config.right_shoulder_camera.depth_in_meters = False
        obs_config.left_shoulder_camera.depth_in_meters = False
        obs_config.overhead_camera.depth_in_meters = False
        obs_config.wrist_camera.depth_in_meters = False
        obs_config.front_camera.depth_in_meters = False

        # We want to save the masks as rgb encodings.
        obs_config.left_shoulder_camera.masks_as_one_channel = False
        obs_config.right_shoulder_camera.masks_as_one_channel = False
        obs_config.overhead_camera.masks_as_one_channel = False
        obs_config.wrist_camera.masks_as_one_channel = False
        obs_config.front_camera.masks_as_one_channel = False

        arm_gripper_mode = self.config.arm_gripper_mode
        if arm_gripper_mode == "joint_positions":
            arm_action_mode = JointPosition(self.config.absolute_mode)
            action_mode = MoveArmThenGripper(
                arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
            )
            env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True,
            )
            self._env = env
            self._env.launch()
            self._task = env.get_task(task_file_to_task_class(game_name))
            self._task.set_variation(self.get_variation())
        elif arm_gripper_mode == "joint_velocities":
            arm_action_mode = JointVelocity()
            action_mode = MoveArmThenGripper(
                arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
            )
            env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True,
            )
            self._env = env
            self._env.launch()
            self._task = env.get_task(task_file_to_task_class(game_name))
            self._task.set_variation(self.get_variation())
        elif arm_gripper_mode == "gripper_pose":
            arm_action_mode = EndEffectorPoseViaPlanning(self.config.absolute_mode)
            action_mode = MoveArmThenGripper(
                arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
            )
            env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True,
            )
            self._env = env
            self._env.launch()
            self._task = env.get_task(task_file_to_task_class(game_name))
            self._task.set_variation(self.get_variation())
        elif arm_gripper_mode == "next_best_pose":
            arm_action_mode = EndEffectorPoseViaPlanning(self.config.absolute_mode)
            action_mode = MoveArmThenGripper(
                arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
            )
            task = task_file_to_task_class(game_name)
            env = KeypointEnvironment(
                task_class=task,
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True,
            )
            self._env = env
            self._env.launch()
            self._task = env._task
            self._task.set_variation(self.get_variation())
        else:
            raise ValueError(f"Unknown arm_gripper_mode: {arm_gripper_mode}")

        _, obs = self._task.reset()
        self._prev_obs = None

        self._task._scene.register_step_callback(self._record_callback)

        if self.config.record_video and arm_gripper_mode == "next_best_pose":
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            cam_base = Dummy("cam_cinematic_base")
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL3)

    def get_variation(self):
        if self.config.variation == -1:
            variation = random.randint(0, self._task.get_variation_count() - 1)
        else:
            variation = self.config.variation
        return variation

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(0, 255, (256, 256, 3), dtype=np.uint8)

    @property
    def action_space(self) -> gym.Space:
        if self.config.arm_gripper_mode == "next_best_pose":
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self._env._rlbench_env.action_shape,
                dtype=np.float32,
            )
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_shape, dtype=np.float32
        )

    def reset(self):
        if self.config.arm_gripper_mode == "next_best_pose":
            obs = self._env.reset()
        else:
            _, obs = self._task.reset()
        self._prev_obs = obs
        res = self.get_image_state(obs)

        self._i = 0
        self._episode_index += 1
        self._record_current_episode = (
            self.config.record_video
            and self._episode_index % self.config.record_every == 0
        )
        self._recorded_images.clear()
        return res

    def _record_callback(self):
        if self._record_current_episode:
            if self.config.arm_gripper_mode == "next_best_pose":
                self._record_cam.handle_explicitly()
                cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
                self._recorded_images.append(cap)
            else:
                img = self._task._scene.get_observation().front_rgb
                self._recorded_images.append(img.astype(np.uint8))

    def _append_final_frame(self):
        if self.config.arm_gripper_mode == "next_best_pose":
            self._record_cam.handle_explicitly()
            img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        else:
            img = self._task._scene.get_observation().front_rgb
        self._recorded_images.append(img)

    def step(self, action: np.ndarray):
        assert np.isfinite(action).all(), action

        try:
            if self.config.arm_gripper_mode == "next_best_pose":
                obs, reward, terminal, success = self._env.step(action)
            else:
                obs, reward, terminal = self._task.step(action)
                success, _ = self._task._task.success()
            self._prev_obs = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            success = False
            reward = 0.0
            obs = self._prev_obs

        res = self.get_image_state(obs)

        self._i += 1

        if terminal or self._i == self.config.episode_length:
            done = True
            if self._record_current_episode:
                self._append_final_frame()
                vid = np.array(self._recorded_images)
            else:
                vid = None
        else:
            done = False
            vid = None

        info = {
            "success": success,
            "vid": vid,
            "episode_len": self._i - 1,
            "terminal": terminal,
        }
        return res, reward, done, info

    def get_image_state(self, obs):
        res = {"image": {}}
        if self.config.arm_gripper_mode == "next_best_pose":
            obs = obs
        else:
            obs = obs.__dict__
        for k in self.config.image_key.split(", "):
            res["image"][k] = obs[k]
        if self.config.state_key != "":
            res["state"] = np.concatenate(
                [obs[k] for k in self.config.state_key.split(", ")]
            )
        return res


if __name__ == "__main__":
    config = RLBenchArmGripper.get_default_config()
    config.arm_gripper_mode = "joint_positions"
    config.state_key = "gripper_pose, gripper_joint_positions"
    config.record_video = True
    config.episode_length = 10
    config.record_every = 1
    env = RLBenchArmGripper("reach_target", config)
    init = env.reset()
    timestep = 0
    for _ in range(100):
        timestep += 1
        res, rew, done, info = env.step(env.action_space.sample())
        print(timestep)
        if done:
            break

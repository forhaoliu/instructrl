from typing import List, Type

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Dummy, VisionSensor
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.backend.utils import task_file_to_task_class
from yarr.agents.agent import ActResult, VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

RECORD_EVERY = 20


class KeypointEnvironment(RLBenchEnv):
    def __init__(
        self,
        task_class,
        action_mode,
        obs_config,
        headless,
    ):
        super(KeypointEnvironment, self).__init__(
            task_class=task_class,
            observation_config=obs_config,
            action_mode=action_mode,
            headless=headless,
            channels_last=True,
        )
        self._previous_obs_dict = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(KeypointEnvironment, self).observation_elements
        for oe in obs_elems:
            if oe.name == "low_dim_state":
                oe.shape = (
                    oe.shape[0] - 7 * 2,
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        obs_elems.append(ObservationElement("gripper_pose", (7,), np.float32))
        # obs_elems.append(
        #     ObservationElement("gripper_joint_positions", (2,), np.float32)
        # )
        # obs_elems.append(ObservationElement("joint_velocities", (7,), np.float32))
        # obs_elems.append(ObservationElement("joint_positions", (7,), np.float32))
        return obs_elems

    def extract_obs(self, obs: Observation):
        joint_vel = obs.joint_velocities
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(KeypointEnvironment, self).extract_obs(obs)

        obs.gripper_matrix = grip_mat
        obs.joint_positions = joint_pos

        obs_dict["gripper_joint_positions"] = obs.gripper_joint_positions
        obs_dict["gripper_pose"] = grip_pose
        obs_dict["joint_velocities"] = joint_vel
        obs_dict["joint_positions"] = joint_pos
        return obs_dict

    def launch(self):
        super(KeypointEnvironment, self).launch()

    def reset(self):
        self._previous_obs_dict = super(KeypointEnvironment, self).reset()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def step(self, action):
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

        return obs, reward, terminal, success


if __name__ == "__main__":
    task = task_file_to_task_class("reach_target")
    env = KeypointEnvironment(
        task, MoveArmThenGripper(JointPosition(), Discrete()), ObservationConfig(), True
    )
    env.launch()
    env._task.set_variation(0)
    obs = env.reset()

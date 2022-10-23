import os
import random
import time
from collections import defaultdict, deque

import h5py
import numpy as np
import rlbench.backend.task as rlbench_task
from absl import app, flags
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from tqdm.auto import tqdm, trange

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "tasks",
    [
        "reach_target",
        "phone_on_base",
        "pick_and_lift",
        "pick_up_cup",
        "put_rubbish_in_bin",
        "stack_wine",
        "take_lid_off_saucepan",
        "take_umbrella_out_of_umbrella_stand",
    ],
    "The tasks to collect. If empty, all tasks are collected.",
)
flags.DEFINE_list("image_size", [256, 256], "The size of the images tp save.")


def create_env():
    img_size = list(map(int, FLAGS.image_size))

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

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
    )
    rlbench_env = Environment(action_mode, obs_config=obs_config, headless=True)
    rlbench_env.launch()

    return rlbench_env


def main(argv):
    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]
    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError(f"Task {t} not recognised!.")
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]
    num_tasks = len(tasks)

    rlbench_env = create_env()

    for task_index in range(num_tasks):
        task = tasks[task_index]
        task_env = rlbench_env.get_task(task)
        var_target = task_env.variation_count()

        # for var in range(var_target):
        #     task_env.set_variation(var)
        print(
            "Task:",
            task_env.get_name(),
            "// Variation:",
            var_target,
        )

    rlbench_env.shutdown()


if __name__ == "__main__":
    app.run(main)

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

from data.utils import convert_keypoints, keypoint_discovery

FLAGS = flags.FLAGS

flags.DEFINE_string("save_path", "./data/keypoint/", "Where to save the demos.")
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
flags.DEFINE_integer(
    "train_episodes_per_task", 10, "The number of episodes to collect per task."
)
flags.DEFINE_integer(
    "val_episodes_per_task", 1, "The number of episodes to collect per task."
)
flags.DEFINE_integer(
    "variations", -1, "Number of variations to collect per task. -1 for all."
)
flags.DEFINE_integer("num_frames", 4, "Number of frames to stack.")
flags.DEFINE_integer("vox_size", 16, "Voxel size to discretize translation.")
flags.DEFINE_integer(
    "rotation_resolution", 5, "Rotation resolution to discretize rotation."
)

# fmt: off
# ['left_shoulder_depth', 'left_shoulder_mask', 'left_shoulder_point_cloud', 'right_shoulder_depth', 'right_shoulder_mask', 'right_shoulder_point_cloud', 'overhead_depth', 'overhead_rgb', 'overhead_mask', 'overhead_point_cloud', 'wrist_depth', 'wrist_mask', 'wrist_point_cloud', 'front_depth', 'front_mask', 'front_point_cloud', 'joint_forces', 'gripper_pose', 'gripper_matrix', 'gripper_joint_positions', 'gripper_touch_forces', 'task_low_dim_state', 'misc',
# 'front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'joint_positions', 'gripper_open', 'joint_velocities']
# fmt: on

TO_BE_REMOVED = ["misc", "task_low_dim_state"]
TO_BE_ADDED = [
    "cont_action",
    "disc_action",
    "time",
    "gripper_pose_delta",
    "task_id",
    "variation_id",
    # "ignore_collisions",
]


def get_shape_dtype(k, dummy_timestep):
    if k == "gripper_open" or k == "time":
        v_dtype = np.float32
        v_shape = (1,)
    elif k == "cont_action":
        v_dtype = np.float32
        v_shape = (8,)
    elif k == "disc_action":
        v_dtype = np.int32
        v_shape = (7,)
    elif k == "gripper_pose_delta":
        v_dtype = np.float32
        v_shape = (7,)
    elif k == "task_id":
        v_dtype = h5py.special_dtype(vlen=np.dtype("uint8"))
        v_shape = ()
    elif k == "variation_id":
        v_dtype = np.uint8
        v_shape = (1,)
    elif k == "ignore_collisions":
        v_dtype = np.uint8
        v_shape = (1,)
    else:
        v_dtype = dummy_timestep.__dict__[k].dtype
        v_shape = dummy_timestep.__dict__[k].shape
    return v_shape, v_dtype


def create_hdf5(rlbench_env, task, hdf5_name, hdf5_shuffled_name, size=int(1e5)):
    dummy_task = rlbench_env.get_task(task)
    dummy_demo = np.array(dummy_task.get_demos(1, live_demos=True)[0])
    dummy_timestep = dummy_demo[0]

    keys = list(dummy_timestep.__dict__.keys())
    keys.extend(TO_BE_ADDED)
    for key in TO_BE_REMOVED:
        keys.remove(key)

    h5_file = h5py.File(hdf5_name, "x")
    h5_file_shuffled = h5py.File(hdf5_shuffled_name, "x")

    for k in keys:
        v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)

        h5_file.create_dataset(
            k,
            (size, FLAGS.num_frames, *v_shape),
            dtype=v_dtype,
            chunks=(16, FLAGS.num_frames, *v_shape),
        )
        h5_file_shuffled.create_dataset(
            k,
            (size, FLAGS.num_frames, *v_shape),
            dtype=v_dtype,
            chunks=(16, FLAGS.num_frames, *v_shape),
        )
    return h5_file, h5_file_shuffled


def collect_data(
    rlbench_env,
    task,
    task_name,
    tasks_with_problems,
    h5_file,
    h5_file_shuffled,
    num_episodes,
):
    variation_count = 0
    total_timestep = 0

    task_env = rlbench_env.get_task(task)

    dummy_task = rlbench_env.get_task(task)
    dummy_demo = np.array(dummy_task.get_demos(1, live_demos=True)[0])
    dummy_timestep = dummy_demo[0]
    keys = list(dummy_timestep.__dict__.keys())
    keys.extend(TO_BE_ADDED)
    for key in TO_BE_REMOVED:
        keys.remove(key)

    total_data = defaultdict(list)

    var_target = task_env.variation_count()
    if FLAGS.variations >= 0:
        var_target = np.minimum(FLAGS.variations, var_target)

    print("Task:", task_env.get_name(), "// Variation Target:", var_target)
    while True:
        if variation_count >= var_target:
            break
        task_env.set_variation(variation_count)
        _, _ = task_env.reset()

        abort_variation = False

        for ex_idx in range(num_episodes):
            print(
                "Task:",
                task_env.get_name(),
                "// Variation:",
                variation_count,
                "// Demo:",
                ex_idx,
            )
            attempts = 10
            while attempts > 0:
                try:
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                    print(f"success. demo length {len(demo)}.")
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        "Failed collecting task %s (variation: %d, "
                        "example: %d). Skipping this task/variation.\n%s\n"
                        % (task_env.get_name(), variation_count, ex_idx, str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break

                def add_more(time, demo, inital_obs, episode_keypoints):
                    obs = inital_obs
                    more_cont_action = []
                    more_disc_action = []
                    more_obs = []
                    more_timestep = []
                    for k, keypoint in enumerate(episode_keypoints):
                        obs_tp1 = demo[keypoint]
                        cont_action, disc_action = convert_keypoints(
                            demo,
                            episode_keypoints,
                            FLAGS.vox_size,
                            FLAGS.rotation_resolution,
                        )
                        more_obs.append(obs)
                        t = (1.0 - time / float(len(demo) - 1)) * 2.0 - 1.0
                        more_timestep.append(t)
                        more_cont_action.append(cont_action)
                        more_disc_action.append(disc_action)
                        obs = obs_tp1
                        t = keypoint
                    return more_obs, more_cont_action, more_disc_action, more_timestep

                episode_keypoints = keypoint_discovery(demo)
                cont_action_list = []
                disc_action_list = []
                obs_list = []
                time_list = []
                for i in range(len(demo) - 1):
                    obs = demo[i]
                    # If our starting point is past one of the keypoints, then remove it
                    while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                        episode_keypoints = episode_keypoints[1:]

                    if len(episode_keypoints) == 0:
                        break

                    cont_action, disc_action = convert_keypoints(
                        demo, episode_keypoints
                    )
                    obs_list.append(obs)
                    cont_action_list.append(cont_action)
                    disc_action_list.append(disc_action)
                    t = (1.0 - i / float(len(demo) - 1)) * 2.0 - 1.0
                    time_list.append(t)

                    (
                        more_obs,
                        more_cont_action,
                        more_disc_action,
                        more_timestep,
                    ) = add_more(i, demo, obs, episode_keypoints)
                    obs_list.extend(more_obs)
                    cont_action_list.extend(more_cont_action)
                    disc_action_list.extend(more_disc_action)
                    time_list.extend(more_timestep)

                stack = defaultdict(lambda: deque([], maxlen=FLAGS.num_frames))
                for idx, timestep in enumerate(obs_list):
                    for k in keys:

                        if k == "gripper_open":
                            v = np.array([timestep.__dict__[k]])
                        elif k == "cont_action":
                            v = cont_action_list[idx]
                        elif k == "disc_action":
                            v = disc_action_list[idx]
                        elif k == "time":
                            v = np.array([time_list[idx]])
                        elif k == "gripper_pose_delta":
                            timestep_tp1 = obs_list[min(idx + 1, len(obs_list) - 1)]
                            v = (
                                timestep_tp1.__dict__["gripper_pose"]
                                - timestep.__dict__["gripper_pose"]
                            )
                        elif k == "task_id":
                            v = np.frombuffer(
                                str(task_name).encode("utf-8"), dtype=np.uint8
                            )
                        elif k == "variation_id":
                            v = np.array([variation_count], dtype=np.uint8)
                        elif k == "ignore_collisions":
                            timestep_tm1 = obs_list[max(0, idx - 1)]
                            v = np.array(
                                [int(timestep_tm1.ignore_collisions)], dtype=np.uint8
                            )
                        else:
                            v = timestep.__dict__[k]
                        if idx == 0:
                            stack[k].extend([v] * FLAGS.num_frames)
                        else:
                            stack[k].append(v)

                    for k in keys:
                        total_data[k].append(np.stack(stack[k]))

                    total_timestep += 1
                break
            if abort_variation:
                break

        variation_count += 1

    start = time.process_time()

    for k in keys:

        v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)

        h5_file[k][:total_timestep] = np.array(total_data[k])

        h5_file[k].resize((total_timestep, FLAGS.num_frames, *v_shape))
        h5_file_shuffled[k].resize((total_timestep, FLAGS.num_frames, *v_shape))

    print(time.process_time() - start)

    indices = list(range(total_timestep))
    random.shuffle(indices)

    for i, j in enumerate(tqdm(indices, desc="shuffling", ncols=0)):
        for k in keys:
            if k == "task_id":
                h5_file_shuffled[k][i] = h5_file[k][j][0]
            else:
                h5_file_shuffled[k][i] = h5_file[k][j]

    h5_file.close()
    h5_file_shuffled.close()


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
    tasks_with_problems = ""

    rlbench_env = create_env()

    for task_index in range(num_tasks):
        task_name = FLAGS.tasks[task_index]

        # train
        train_hdf5_name = os.path.join(FLAGS.save_path, f"{task_name}_train.hdf5")
        train_hdf5_shuffled_name = os.path.join(
            FLAGS.save_path, f"{task_name}_train_shuffled.hdf5"
        )
        try:
            os.remove(train_hdf5_name)
            os.remove(train_hdf5_shuffled_name)
        except OSError:
            pass

        h5_file, h5_file_shuffled = create_hdf5(
            rlbench_env,
            tasks[task_index],
            train_hdf5_name,
            train_hdf5_shuffled_name,
        )
        collect_data(
            rlbench_env,
            tasks[task_index],
            task_name,
            tasks_with_problems,
            h5_file,
            h5_file_shuffled,
            num_episodes=FLAGS.train_episodes_per_task,
        )

        # val
        val_hdf5_name = os.path.join(FLAGS.save_path, f"{task_name}_val.hdf5")
        val_hdf5_shuffled_name = os.path.join(
            FLAGS.save_path, f"{task_name}_val_shuffled.hdf5"
        )
        try:
            os.remove(val_hdf5_name)
            os.remove(val_hdf5_shuffled_name)
        except OSError:
            pass

        h5_file, h5_file_shuffled = create_hdf5(
            rlbench_env,
            tasks[task_index],
            val_hdf5_name,
            val_hdf5_shuffled_name,
        )
        collect_data(
            rlbench_env,
            tasks[task_index],
            task_name,
            tasks_with_problems,
            h5_file,
            h5_file_shuffled,
            num_episodes=FLAGS.val_episodes_per_task,
        )

    print(tasks_with_problems)

    combine_multi_task(rlbench_env, tasks, num_tasks)

    rlbench_env.shutdown()

    test(argv)


def combine_multi_task(rlbench_env, tasks, num_tasks):

    dummy_task = rlbench_env.get_task(tasks[0])
    dummy_demo = np.array(dummy_task.get_demos(1, live_demos=True)[0])
    dummy_timestep = dummy_demo[0]
    keys = list(dummy_timestep.__dict__.keys())
    keys.extend(TO_BE_ADDED)
    for key in TO_BE_REMOVED:
        keys.remove(key)

    combine_hdf5_files(
        rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="train"
    )

    combine_hdf5_files(rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="val")


def combine_hdf5_files(
    rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="train"
):
    print("combining hdf5 files for", split)

    total_data_fn = os.path.join(FLAGS.save_path, f"multi_task_{split}.hdf5")
    total_data_shuffled_fn = os.path.join(
        FLAGS.save_path, f"multi_task_{split}_shuffled.hdf5"
    )

    try:
        os.remove(total_data_fn)
        os.remove(total_data_shuffled_fn)
    except OSError:
        pass

    total_data, total_data_shuffled = create_hdf5(
        rlbench_env,
        tasks[0],
        total_data_fn,
        total_data_shuffled_fn,
        size=int(1e5) * num_tasks,
    )

    total_timestep = 0
    for task_index in trange(num_tasks, desc="combining data", ncols=0):
        task_name = FLAGS.tasks[task_index]

        h5_file_name = os.path.join(
            FLAGS.save_path, f"{task_name}_{split}_shuffled.hdf5"
        )
        h5_file = h5py.File(h5_file_name, "r")
        h5_size = h5_file[keys[0]].shape[0]

        for k in keys:
            total_data[k][total_timestep : total_timestep + h5_size] = h5_file[k][:]

        total_timestep += h5_file[k].shape[0]

    for k in keys:

        v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)

        total_data[k].resize((total_timestep, FLAGS.num_frames, *v_shape))
        total_data_shuffled[k].resize((total_timestep, FLAGS.num_frames, *v_shape))

    indices = list(range(total_timestep))
    random.shuffle(indices)

    for i, j in enumerate(tqdm(indices, desc="shuffling", ncols=0)):
        for k in keys:
            if k == "task_id":
                total_data_shuffled[k][i] = total_data[k][j][0]
            else:
                total_data_shuffled[k][i] = total_data[k][j]


def test(argv):
    h5_file = h5py.File(
        os.path.join(FLAGS.save_path, "multi_task_train_shuffled.hdf5"), "r"
    )
    for k in [
        "joint_velocities",
        "joint_positions",
        "gripper_pose",
        "gripper_joint_positions",
        "gripper_pose_delta",
    ]:
        print(k, h5_file[k].shape)
    for k in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]:
        print(k, h5_file[k].shape)


if __name__ == "__main__":
    app.run(main)  # run and test
    # app.run(test) # test only

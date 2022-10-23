import threading
from io import BytesIO
from queue import Queue

import gcsfs
import h5py
import numpy as np
import pyrender
import skimage.io
import torch
import trimesh
from ml_collections import ConfigDict
from PIL import Image
from pyrender.trackball import Trackball
from rlbench.backend.const import DEPTH_SCALE
from rlbench.const import colors
from scipy.spatial.transform import Rotation
from skimage.color import gray2rgb, rgba2rgb


class RLBenchDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = "gs://instruct-rl/dataset/variation"

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_size = 256

        config.state_key = ""
        config.state_dim = 0

        config.image_key = "front_rgb, left_shoulder_rgb, right_shoulder_rgb, wrist_rgb"

        config.action_key = "gripper_pose, gripper_open, ignore_collisions"
        config.action_dim = 7 + 1 + 1

        config.need_quat_normalize = "gripper_pose, gripper_pose_delta"

        config.use_bert_tokenizer = True
        config.tokenizer_max_length = 77

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(
        self,
        update,
        dataset_name="reach_target",
        start_offset_ratio=None,
        split="train",
    ):
        self.config = self.get_default_config(update)
        assert self.config.path != ""

        self.dataset_name = dataset_name

        if split == "train":
            path = f"{self.config.path}/{dataset_name}_train_shuffled.hdf5"
        elif split == "val":
            path = f"{self.config.path}/{dataset_name}_val_shuffled.hdf5"
        else:
            raise ValueError(f"Unknown split: {split}")
        if self.config.path.startswith("gs://"):
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(path, "r")

        if self.config.random_start:
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

        self.tokenizer = self.build_tokenizer()

    def __getstate__(self):
        return self.config, self.random_start_offset, self.dataset_name

    def __setstate__(self, state):
        config, random_start_offset, dataset_name = state
        self.__init__(config)
        self.random_start_offset = random_start_offset
        self.dataset_name = dataset_name

    def __len__(self):
        return min(
            self.h5_file["time"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, index):
        index = self.process_index(index)
        res = {"image": {}}
        for key in self.config.image_key.split(", "):
            res["image"][key] = self.h5_file[key][index]
        if self.config.state_key != "":
            res["state"] = np.concatenate(
                [self.h5_file[k][index] for k in self.config.state_key.split(", ")],
                axis=-1,
            )
        res["action"] = np.concatenate(
            [
                self._normalize_quat(k, self.h5_file[k][index])
                for k in self.config.action_key.split(", ")
            ],
            axis=-1,
        )
        with BytesIO(self.h5_file["task_id"][index][0]) as fin:
            task_id = fin.read().decode("utf-8")
        variation_id = self.h5_file["variation_id"][index][0].item()
        instruct = get_instruct(task_id, variation_id)

        tokenized_instruct, padding_mask = self.tokenizer(instruct)

        res["instruct"] = tokenized_instruct
        res["padding_mask"] = padding_mask

        return res

    def _normalize_quat(self, name, quat):

        if name in self.config.need_quat_normalize.split(", "):
            return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
        else:
            return quat

    def build_tokenizer(self):
        use_bert_tokenizer = self.config.use_bert_tokenizer
        tokenizer_max_length = self.config.tokenizer_max_length

        if use_bert_tokenizer:
            import transformers

            tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            from .models.openai import tokenizer

            tokenizer = tokenizer.build_tokenizer()

        def tokenizer_fn(instruct):
            if use_bert_tokenizer:
                if len(instruct) == 0:
                    tokenized_instruct = np.zeros(tokenizer_max_length, dtype=np.int32)
                    padding_mask = np.ones(tokenizer_max_length, dtype=np.float32)
                else:
                    encoded_instruct = tokenizer(
                        instruct,
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer_max_length,
                        return_tensors="np",
                        add_special_tokens=False,
                    )
                    tokenized_instruct = encoded_instruct["input_ids"][0].astype(
                        np.long
                    )
                    padding_mask = 1.0 - encoded_instruct["attention_mask"][0].astype(
                        np.float32
                    )
            else:
                tokenized_instruct = tokenizer(instruct)[0].astype(np.long)
                padding_mask = None
            return tokenized_instruct, padding_mask

        return tokenizer_fn

    @property
    def num_actions(self):
        return self.config.action_dim

    @property
    def obs_shape(self):
        res = {"image": {}}
        for key in self.config.image_key.split(", "):
            res["image"][key] = (256, 256, 3)
        if self.config.state_key != "":
            res["state"] = self.config.state_dim
        return res


def get_instruct(task, variation):
    # https://github.com/stepjam/RLBench/tree/master/rlbench/tasks
    if task == "stack_wine":
        return f"place the wine bottle on the wine rack."
    elif task == "take_umbrella_out_of_umbrella_stand":
        return f"grasping the umbrella by its handle, lift it up and out of the stand."
    elif task == "reach_target":
        return f"touch the {colors[variation]} ball with the panda gripper."
    elif task == "pick_and_lift":
        return f"pick up the {colors[variation]} block and lift it up to the target."
    elif task == "pick_up_cup":
        return f"grasp the {colors[variation]} cup and lift it."
    elif task == "put_rubbish_in_bin":
        return f"pick up the rubbish and leave it in the trash can."
    elif task == "take_lid_off_saucepan":
        return "grip the saucepan's lid and remove it from the pan."
    elif task == "open_drawer":
        OPTIONS = ["bottom", "middle", "top"]
        return f"grip the {OPTIONS[variation]} handle and pull the {OPTIONS[variation]} drawer open."
    elif task == "meat_off_grill":
        OPTIONS = ["chicken", "steak"]
        return f"pick up the {OPTIONS[variation]} and place it next to the grill."
    elif task == "put_item_in_drawer":
        OPTIONS = ["bottom", "middle", "top"]
        return f"open the {OPTIONS[variation]} drawer and place the block inside of it."
    elif task == "turn_tap":
        OPTIONS = ["left", "right"]
        return "grasp the {OPTIONS[variation]} tap and turn it."
    elif task == "put_groceries_in_cupboard":
        GROCERY_NAMES = [
            "crackers",
            "chocolate jello",
            "strawberry jello",
            "soup",
            "tuna",
            "spam",
            "coffee",
            "mustard",
            "sugar",
        ]
        return f"pick up the {GROCERY_NAMES[variation]} and place it in the cupboard."
    elif task == "place_shape_in_shape_sorter":
        SHAPE_NAMES = ["cube", "cylinder", "triangular prism", "star", "moon"]
        return f"pick up the {SHAPE_NAMES[variation]} and put it in the sorter."
    elif task == "light_bulb_in":
        return f"pick up the light bulb from the {colors[variation]} stand, lift it up to just above the lamp, then screw it down into the lamp in a clockwise fashion."
    elif task == "close_jar":
        return f"close the {colors[variation]} jar."
    elif task == "stack_blocks":
        MAX_STACKED_BLOCKS = 3
        color_index = int(variation / MAX_STACKED_BLOCKS)
        blocks_to_stack = 2 + variation % MAX_STACKED_BLOCKS
        color_name, color_rgb = colors[color_index]
        return (
            f"place {blocks_to_stack} of the {color_name} cubes on top of each other."
        )
    elif task == "put_in_safe":
        safe = {0: "bottom", 1: "middle", 2: "top"}
        return f"put the money away in the safe on the {safe[variation]} shelf."
    # elif task == "push_buttons":
    #     pass
    elif task == "insert_onto_square_peg":
        return f"put the ring on the {colors[variation]} spoke."
    elif task == "stack_cups":
        target_color_name, target_rgb = colors[variation]
        return f"stack the other cups on top of the {target_color_name} cup."
    elif task == "place_cups":  # 3 variations
        return f"place {variation + 1} cups on the cup holder."
    else:
        raise ValueError(f"Unknown task: {task}")


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def get_cont_action(
    disc_action,
    vox_size=100,
    rotation_resolution=5,
    scene_bound=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
):
    coords = disc_action[:3]
    rot = disc_action[3:-1]
    grip = disc_action[-1]

    bound = np.array(scene_bound)

    res = (bound[3:] - bound[:3]) / vox_size
    trans = bound[:3] + res * coords + res / 2

    cont_action = np.concatenate(
        [trans, discrete_euler_to_quaternion(rot, rotation_resolution), grip]
    )
    # translation [0, vox_size - 1]
    # rotation [0, 360 / vox_size - 1]
    return cont_action

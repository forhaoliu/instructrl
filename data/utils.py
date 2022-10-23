import logging
from typing import List

import numpy as np
import pyrender
import torch
import trimesh
from pyrender.trackball import Trackball
from rlbench.backend.const import DEPTH_SCALE
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation

SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if "sampling_probabilities" in replay_sample:
        probs = replay_sample["sampling_probabilities"]
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy


def point_to_pixel_index(
    point: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {
        name: pyrender.Mesh.from_trimesh(geom, smooth=False)
        for name, geom in trimesh_scene.geometry.items()
    }
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([w, w, l], T, face_colors=[0, 0, 0, 255])
        )
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([l, w, w], T, face_colors=[0, 0, 0, 255])
        )
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([w, l, w], T, face_colors=[0, 0, 0, 255])
        )


def create_voxel_scene(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    voxel_size: float = 0.1,
    show_bb: bool = False,
    alpha: float = 0.5,
):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1) / 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = q > 0.75
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate(
            [q, np.zeros_like(q), np.zeros_like(q), np.clip(q, 0, 1)], axis=-1
        )
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0)
    )
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform
    )
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    rotation_amount: float = 0.0,
    show: bool = False,
    voxel_size: float = 0.1,
    offscreen_renderer: pyrender.OffscreenRenderer = None,
    show_bb: bool = False,
):
    scene = create_voxel_scene(
        voxel_grid,
        q_attention,
        highlight_coordinate,
        highlight_alpha,
        voxel_size,
        show_bb,
    )
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0
        )
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8], bg_color=[1.0, 1.0, 1.0]
        )
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height
        )
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)
        color, depth = r.render(s)
        return color.copy()


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    logging.debug("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _get_disc_action(
    obs_tp1,
    vox_size=100,
    rotation_resolution=5,
    scene_bound=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    trans_indicies = []
    bound = np.array(scene_bound)
    index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bound)
    trans_indicies.extend(index.tolist())

    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    action = np.concatenate((np.array(trans_indicies), np.array(rot_and_grip_indicies)))
    return action


def _get_cont_action(obs_tp1):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate(
        [obs_tp1.gripper_pose[:3], quat, [float(obs_tp1.gripper_open)]]
    )


def convert_keypoints(demo, episode_keypoints, vox_size=100, rotation_resolution=5):
    next_keypoint = episode_keypoints[0]
    obs_tp = demo[next_keypoint]
    cont_action = _get_cont_action(obs_tp)
    disc_action = _get_disc_action(
        obs_tp, vox_size=vox_size, rotation_resolution=rotation_resolution
    )
    return cont_action, disc_action

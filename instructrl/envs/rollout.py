import copy
import logging
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange


def batch_rollout(
    rng,
    env,
    policy_fn,
    transform_action_fn,
    episode_length=2500,
    log_interval=None,
    window_size=0,
    num_episodes=1,
    num_actions=8,
):
    all_inputs = {}

    concat_fn = lambda x, y: jnp.concatenate([x, y], axis=0)
    trim_fn = lambda x: x[:, -window_size:, ...]
    batch_fn = lambda x: x[None, None, ...]

    def prepare_input(all_inputs, obs):
        action = jnp.zeros(num_actions)
        inputs = {**obs, "action": action}
        inputs = jax.tree_util.tree_map(batch_fn, inputs)

        if len(all_inputs) == 0:
            inputs = inputs
        else:
            all_inputs_copy = copy.deepcopy(all_inputs)

            inputs = jax.tree_util.tree_map(concat_fn, all_inputs_copy, inputs)
            inputs = jax.tree_util.tree_map(trim_fn, inputs)

        return all_inputs, inputs

    def update_input(all_inputs, obs, action):
        inputs = {**obs, "action": action}
        inputs = jax.tree_util.tree_map(batch_fn, inputs)
        if len(all_inputs) == 0:
            all_inputs = inputs
        else:
            all_inputs = jax.tree_util.tree_map(concat_fn, all_inputs, inputs)
            all_inputs = jax.tree_util.tree_map(trim_fn, all_inputs)

        return all_inputs

    reward = jnp.zeros(1, dtype=jnp.float32)
    success = jnp.zeros(1, dtype=jnp.float32)

    for _ in trange(num_episodes, desc="rollout", ncols=0):

        done = jnp.zeros(1, dtype=jnp.int32)

        for t in trange(episode_length, desc=f"episode {_}", ncols=0, leave=False):

            done_prev = done

            if t == 0:
                obs = env.reset()
            else:
                obs = next_obs

            all_inputs, inputs = prepare_input(all_inputs, obs)
            action = jax.device_get(policy_fn(inputs=inputs, rngs=rng))[0]
            action = transform_action_fn(action)
            all_inputs = update_input(all_inputs, obs, action)

            next_obs, reward, done, info = env.step(action)

            reward = reward + reward * (1 - done_prev)
            done = jnp.logical_or(done, done_prev).astype(jnp.int32)
            success += jnp.array([info["success"]], dtype=jnp.float32)

            if log_interval and t % log_interval == 0:
                logging.info("step: %d done: %s reward: %s", t, done, reward)

            if jnp.all(done):
                break

    metric = {
        "return": reward.astype(jnp.float32) / num_episodes,
        "success": (success > 1).astype(jnp.float32) / num_episodes,
    }
    return metric, info

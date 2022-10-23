import dataclasses
import os
import pprint
from functools import partial
from typing import Any, Callable, Optional

import absl.app
import absl.flags
import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
from absl import app, flags, logging
from flax import jax_utils
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training import checkpoints, common_utils, train_state
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

from .envs import rollout
from .model import BC
from .utils import (
    JaxRNG,
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    load_pickle,
    next_rng,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    load_checkpoint="",
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    model=BC.get_default_config(),
    window_size=4,
    episode_length=500,
    instruct="",
    dataset_name="reach_target",
    num_test_episodes=5,
    num_actions=8,
    obs_shape=(256, 256, 3),
)
FLAGS = absl.flags.FLAGS


def build_env_fn(dataset_name):
    def env_fn():
        from .envs import rlbench

        env = rlbench.create_environment(dataset_name)
        return env

    return env_fn


def create_test_step(
    model, env_fn, episode_length, instruct, window_size, num_episodes, num_actions
):
    @jax.jit
    def policy_fn(variables, inputs, rngs):
        inputs.update(instruct)
        output = model.apply(
            variables,
            inputs,
            rngs=rngs,
            method=model.greedy_action,
        )
        return output

    def test_step_fn(state, rng):
        next_rng, split_rng = jax.random.split(rng)
        rng_generator = JaxRNG(split_rng)
        policy = partial(policy_fn, variables={"params": state.params})
        aux = rollout.batch_rollout(
            rng_generator(model.rng_keys()),
            env_fn,
            policy,
            episode_length,
            window_size=window_size,
            num_episodes=num_episodes,
            num_actions=num_actions,
        )
        return aux, next_rng

    return test_step_fn


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    model = BC(
        config_updates=FLAGS.model,
        num_actions=FLAGS.num_actions,
        obs_shape=FLAGS.obs_shape,
        patch_dim=16,
    )

    def tokenize_fn(text):
        if FLAGS.model.transfer_type.startswith("clip"):
            from .models.openai import tokenizer

            token_fn = tokenizer.build_tokenizer()
            tokenized_text = token_fn(text).astype(np.long)
        elif FLAGS.model.transfer_type.startswith("m3ae"):
            import transformers

            tokenizer = partial(
                transformers.BertTokenizer.from_pretrained("bert-base-uncased"),
                truncation=True,
                return_tensors="np",
                add_special_tokens=False,
            )
            tokenized_text = tokenizer(text)["input_ids"].astype(np.long)
        else:
            assert (
                False
            ), f"{FLAGS.instruct} not supported with {FLAGS.model.transfer_type}"
        return tokenized_text

    test_instruct = {"instruct": None}
    if FLAGS.instruct != "":
        instruct_token = tokenize_fn(FLAGS.instruct)
        test_instruct = {"instruct": instruct_token}

    assert FLAGS.load_checkpoint != "", "load_checkpoint is required"
    checkpoint_data = load_pickle(FLAGS.load_checkpoint)
    state = checkpoint_data["state"]

    env_fn = build_env_fn(FLAGS.dataset_name)
    test_step_fn = create_test_step(
        model,
        env_fn(),
        FLAGS.episode_length,
        test_instruct,
        FLAGS.window_size,
        FLAGS.num_test_episodes,
        FLAGS.num_actions,
    )

    log_metrics, _ = test_step_fn(state, next_rng())
    log_metrics = {
        f"test_{k}": v for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
    }
    logger.log(log_metrics)
    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == "__main__":
    jax.config.config_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)

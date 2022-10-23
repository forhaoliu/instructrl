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
import wandb
from absl import app, flags, logging
from flax import jax_utils
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training import checkpoints, common_utils, train_state
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

from .data import RLBenchDataset, get_cont_action, get_instruct
from .envs import rollout
from .envs.rlbench import RLBenchArmGripper
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
    epochs=100,
    warmup_epochs=5.0,
    weight_decay=1e-4,
    batch_size=2,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=100,
    save_model_freq=0,
    load_checkpoint="",
    lr=0.1,
    lr_schedule="cos",
    momentum=0.9,
    clip_gradient=1e9,
    auto_scale_lr=False,
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    model=BC.get_default_config(),
    data=RLBenchDataset.get_default_config(),
    dataset_name="reach_target",
    env=RLBenchArmGripper.get_default_config(),
    window_size=4,
    instruct="",
    val_every_epochs=10,
    test_every_epochs=10,
    num_test_episodes=5,
    game_name="reach_target",
    is_tpu=False,
    tokenizer_max_length=77,
)
FLAGS = absl.flags.FLAGS


def build_env_fn(game_name):
    def env_fn():
        env = RLBenchArmGripper(game_name, FLAGS.env)
        return env

    return env_fn


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_fn(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_map(select, state)


def create_train_step(model, learning_rate, weight_decay):
    def loss_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)
        output = model.apply(
            {"params": params},
            batch,
            rngs=rng_generator(model.rng_keys()),
            deterministic=False,
        )
        loss = output["loss"]
        weight_penalty_params = jax.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        aux = dict(
            loss=loss,
            weight_penalty=weight_penalty,
            weight_l2=weight_l2,
        )
        return loss, (aux,)

    @partial(jax.pmap, axis_name="pmap", donate_argnums=(0))
    def train_step_fn(state, batch, rng):
        next_rng, split_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (aux,)), grads = jax.lax.pmean(
            grad_fn(state.params, batch, split_rng), axis_name="pmap"
        )

        aux["train_state_step"] = state.step
        aux["learning_rate"] = learning_rate(state.step)

        new_state = state.apply_gradients(grads=grads)

        return new_state, aux, next_rng

    return train_step_fn


def create_val_step(model):
    def val_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)
        output = model.apply(
            {"params": params},
            batch,
            rngs=rng_generator(model.rng_keys()),
            deterministic=True,
        )
        aux = dict(loss=output["loss"])

        return aux

    @partial(jax.pmap, axis_name="pmap")
    def val_step_fn(state, batch, rng):
        next_rng, split_rng = jax.random.split(rng)
        aux = jax.lax.pmean(val_fn(state.params, batch, split_rng), axis_name="pmap")
        return aux, next_rng

    return val_step_fn


def create_test_step(
    model,
    environment,
    episode_length,
    instruct,
    window_size,
    num_episodes,
    num_actions,
    transform_action_fn,
):
    @jax.jit
    def policy_fn(variables, inputs, rngs):
        inputs.update(instruct)
        output = model.apply(
            variables=variables,
            batch=inputs,
            rngs=rngs,
            method=model.greedy_action,
        )
        return output

    def test_step_fn(state, rng):
        next_rng, split_rng = jax.random.split(rng)
        rng_generator = JaxRNG(split_rng)
        policy = partial(policy_fn, variables={"params": state.params})
        metric, info = rollout.batch_rollout(
            rng_generator(model.rng_keys()),
            environment,
            policy,
            episode_length,
            transform_action_fn,
            window_size=window_size,
            num_episodes=num_episodes,
            num_actions=num_actions,
        )
        return metric, info, next_rng

    return test_step_fn


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    assert FLAGS.batch_size % jax_process_count == 0
    variant["process_batch_size"] = process_batch_size = (
        FLAGS.batch_size // jax_process_count
    )
    variant["device_batch_size"] = device_batch_size = (
        process_batch_size // jax.local_device_count()
    )
    if FLAGS.auto_scale_lr:
        lr_scale = FLAGS.batch_size / 256
    else:
        lr_scale = 1.0
    variant["effective_lr"] = FLAGS.lr * lr_scale
    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert process_batch_size % n_devices == 0

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    train_dataset = RLBenchDataset(
        FLAGS.data,
        FLAGS.dataset_name,
        jax_process_index / jax_process_count,
        split="train",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )
    val_dataset = RLBenchDataset(
        FLAGS.data,
        FLAGS.dataset_name,
        jax_process_index / jax_process_count,
        split="val",
    )
    val_batch_size = min(process_batch_size, len(val_dataset) // jax_process_count)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    steps_per_epoch = int(len(train_dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs
    val_steps = int(len(val_dataset) / val_batch_size)

    if FLAGS.save_model_freq > 0:
        save_model_freq = FLAGS.save_model_freq
    else:
        save_model_freq = steps_per_epoch * 10

    normalize_quterion = FLAGS.env.arm_gripper_mode in [
        "next_best_pose",
        "gripper_pose",
    ]
    model = BC(
        config_updates=FLAGS.model,
        num_actions=train_dataset.num_actions,
        patch_dim=16,
        normalize_quterion=normalize_quterion,
    )

    if FLAGS.lr_schedule == "fixed":
        learning_rate = lambda x: FLAGS.lr * lr_scale
    elif FLAGS.lr_schedule == "cos":
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=FLAGS.lr * lr_scale,
            warmup_steps=int(FLAGS.warmup_epochs * steps_per_epoch),
            decay_steps=total_steps,
            end_value=0.0,
        )
    else:
        raise ValueError("Unsupported lr schedule!")

    def get_dummy_input():
        dummy_input = {
            "action": jnp.ones((1, 4, train_dataset.num_actions), dtype=jnp.float32),
        }
        if train_dataset.config.state_key != "":
            dummy_input["state"] = jnp.ones(
                (1, 4, train_dataset.config.state_dim), dtype=jnp.float32
            )
        dummy_input["image"] = {}
        for k, v in train_dataset.obs_shape["image"].items():
            dummy_input["image"][k] = jnp.ones((1, 4, *v), dtype=jnp.float32)

        dummy_instruct = {
            "instruct": jnp.zeros((1, 4, FLAGS.tokenizer_max_length), dtype=jnp.int32)
        }
        dummy_input.update(dummy_instruct)

        return dummy_input

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = jax_utils.replicate(checkpoint_data["state"], jax_devices)
        start_step = checkpoint_data["step"]
    else:

        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)

        dummy_input = get_dummy_input()
        params = init(next_rng(model.rng_keys()), dummy_input, deterministic=False)[
            "params"
        ]
        params = flax.core.frozen_dict.unfreeze(params)

        def get_weight_decay_mask(params):
            flattened_params = flax.traverse_util.flatten_dict(
                flax.core.frozen_dict.unfreeze(params)
            )

            def decay(key):
                return all([k not in model.no_decay_list() for k in key])

            return flax.traverse_util.unflatten_dict(
                {key: decay(key) for key in flattened_params.keys()}
            )

        tx = optax.chain(
            optax.clip_by_global_norm(FLAGS.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=FLAGS.weight_decay,
                b1=0.9,
                b2=0.999,
                mask=get_weight_decay_mask,
            ),
        )

        state = jax_utils.replicate(
            TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx,
            ),
            jax_devices,
        )
        start_step = 0

    def flops(params):
        f = lambda x: model.apply({"params": flax.core.freeze(params)}, x)
        xla_f = jax.xla_computation(f)
        dummy_input = get_dummy_input()
        computation = xla_f(dummy_input)
        module = computation.as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
        return analysis

    if jax.process_index() == 0:
        analysis = flops(jax_utils.unreplicate(state.params))
        logging.info(f"flops: {analysis['flops']}")
        logger.log({"cost/flops": analysis["flops"]})
        num_params = sum(
            p.size for p in jax.tree_leaves(jax_utils.unreplicate(state.params))
        )
        logging.info(f"num_params: {num_params}")
        logger.log({"cost/num_params": num_params})

    train_step_fn = create_train_step(model, learning_rate, FLAGS.weight_decay)
    val_step_fn = create_val_step(model)
    if not FLAGS.is_tpu:
        environment = build_env_fn(FLAGS.game_name)()
        tokenizer = train_dataset.build_tokenizer()
        test_instruct, test_padding_mask = tokenizer(get_instruct(FLAGS.game_name, 0))
        instruct_info = {"instruct": test_instruct, "padding_mask": test_padding_mask}
        if FLAGS.model.use_discrete_action:
            transform_action_fn = partial(
                get_cont_action,
                vox_size=FLAGS.model.vox_size,
                rotation_resolution=FLAGS.model.rotation_resolution,
                scene_bound=FLAGS.model.scene_bound,
            )
        else:
            transform_action_fn = lambda x: x
        test_step_fn = create_test_step(
            model,
            environment,
            FLAGS.env.episode_length,
            instruct_info,
            FLAGS.window_size,
            FLAGS.num_test_episodes,
            train_dataset.num_actions,
            transform_action_fn,
        )

    state = sync_state_fn(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    def generate_batch(iterator):
        while True:
            for batch in iterator:

                reshape_fn = lambda x: x.numpy().reshape(n_devices, -1, *x.shape[1:])

                image = jax.tree_util.tree_map(reshape_fn, batch["image"])
                if "state" in batch:
                    state = jax.tree_util.tree_map(reshape_fn, batch["state"])
                else:
                    state = None
                action = jax.tree_util.tree_map(reshape_fn, batch["action"])
                if batch["instruct"] is not None:
                    instruct = jax.tree_util.tree_map(reshape_fn, batch["instruct"])
                else:
                    instruct = None
                if batch["padding_mask"] is not None:
                    padding_mask = jax.tree_util.tree_map(
                        reshape_fn, batch["padding_mask"]
                    )
                else:
                    padding_mask = None

                yield {
                    "image": image,
                    "state": state,
                    "action": action,
                    "instruct": instruct,
                    "padding_mask": padding_mask,
                }

    train_iter = prefetch_to_device(generate_batch(train_loader), 2, jax_devices)
    val_iter = prefetch_to_device(generate_batch(val_loader), 2, jax_devices)
    step_counter = trange(start_step, total_steps, desc="Train...", ncols=0)

    for step, batch in zip(step_counter, train_iter):
        if step % steps_per_epoch == 0:
            train_metrics = []

        epoch = step // steps_per_epoch

        state, metrics, sharded_rng = train_step_fn(state, batch, sharded_rng)
        train_metrics.append(metrics)

        if step % FLAGS.log_freq == 0:
            log_metrics = common_utils.get_metrics(train_metrics)
            log_metrics = {
                f"train_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if step % save_model_freq == 0 or step == total_steps - 1:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(flax.jax_utils.unreplicate(state)),
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")

        if (
            FLAGS.val_every_epochs > 0
            and step % (FLAGS.val_every_epochs * steps_per_epoch) == 0
        ):
            val_metrics = []
            for _, batch in zip(trange(val_steps, desc="val...", ncols=0), val_iter):
                metrics, _ = val_step_fn(state, batch, sharded_rng)
                val_metrics.append(metrics)

            log_metrics = common_utils.get_metrics(val_metrics)
            log_metrics = {
                f"val_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if (
            not FLAGS.is_tpu
            and FLAGS.test_every_epochs > 0
            and step % (FLAGS.test_every_epochs * steps_per_epoch) == 0
        ):
            log_metrics, log_infos, _ = test_step_fn(
                flax.jax_utils.unreplicate(state), next_rng()
            )
            log_metrics = {
                f"test_{k}": v
                for k, v in jax.tree_map(
                    lambda x: jax.device_get(x)[0], log_metrics
                ).items()
            }
            log_metrics.update(
                {"step": step, "epoch": epoch, "episode_len": log_infos["episode_len"]}
            )
            logger.log(log_metrics)
            if log_infos["vid"] is not None:
                frames = np.transpose(log_infos["vid"], (0, 3, 1, 2))
                fps, skip = 2, 1
                if log_infos["vid"].shape[0] > 1:
                    logger.log(
                        {
                            "media/video": wandb.Video(
                                frames[::skip, :, :, :], fps=fps, format="gif"
                            )
                        }
                    )
                else:
                    logger.log(
                        {
                            "media/image": wandb.Video(
                                frames[::skip, :, :, :], fps=fps, format="gif"
                            )
                        }
                    )
                logger.log(
                    {
                        "media/step": step,
                        "media/epoch": epoch,
                        "media/episode_len": log_infos["episode_len"],
                    }
                )
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == "__main__":
    jax.config.config_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)

from pogema import pogema_v0, AnimationMonitor

import torch

from magat_plus.runtime_data_generation import get_runtime_data_generator
from magat_plus.collision_shielding import get_collision_shielded_model


@torch.no_grad()
def run_model_on_grid(
    model,
    device,
    grid_config,
    args,
    dataset_kwargs,
    max_episodes=None,
    aux_func=None,
    animation_monitor=False,
):
    env = pogema_v0(grid_config=grid_config)
    if animation_monitor:
        env = AnimationMonitor(env)
    observations, infos = env.reset()

    model.in_simulation(True)

    rt_data_generator = get_runtime_data_generator(
        grid_config=grid_config,
        args=args,
        dataset_kwargs=dataset_kwargs,
    )

    if aux_func is not None:
        aux_func(
            env=env, observations=observations, actions=None, rtdg=rt_data_generator
        )

    model = get_collision_shielded_model(
        model, env, args, rt_data_generator=rt_data_generator
    )

    while True:
        actions = model.get_actions(observations)
        observations, rewards, terminated, truncated, infos = env.step(actions)

        if aux_func is not None:
            aux_func(
                env=env,
                observations=observations,
                actions=actions,
                rtdg=rt_data_generator,
            )

        if all(terminated) or all(truncated):
            break

        if max_episodes is not None:
            max_episodes -= 1
            if max_episodes <= 0:
                break
    model.in_simulation(False)
    return all(terminated), env, observations

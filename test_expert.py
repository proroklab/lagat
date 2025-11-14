import argparse
import pickle
import pathlib
import numpy as np
import wandb
import signal
from contextlib import contextmanager

from pogema import pogema_v0, GridConfig

from magat_plus.run_expert import get_expert_algorithm_and_config, add_lagat_args
from grid_config_generator import add_grid_config_args, grid_config_generator_factory


def add_expert_dataset_args(parser):
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser = add_grid_config_args(parser)

    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

    parser.add_argument(
        "--save_termination_state", action=argparse.BooleanOptionalAction, default=False
    )

    parser = add_lagat_args(parser)

    return parser


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def run_expert_algorithm_time_limit(
    expert, env=None, observations=None, grid_config=None, time_limit=None
):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    makespan = 0
    costs = np.ones(env.get_num_agents())
    loses = np.ones(env.get_num_agents())

    expert.reset_states(env)

    terminated = [False] * env.get_num_agents()

    try:
        with time_limit_run(time_limit):
            while True:
                actions = expert.act(observations)

                original_pos = np.array([obs["global_xy"] for obs in observations])

                observations, rewards, terminated, truncated, infos = env.step(actions)

                new_pos = np.array([obs["global_xy"] for obs in observations])
                at_goals = np.array(env.was_on_goal)

                makespan += 1
                costs[~at_goals] = makespan + 1
                loses[~at_goals] += 1

                if all(terminated) or all(truncated):
                    break
    except TimeoutException as e:
        print("Timed out")
        # If we hit the time limit, we consider stay actions for all agents
        at_goals = np.array(env.was_on_goal)
        if not all(at_goals):
            if np.max(costs) != makespan + 1:
                costs[~at_goals] = makespan + 1
                loses[~at_goals] += 1
            loses[~at_goals] += grid_config.max_episode_steps - makespan
            makespan = grid_config.max_episode_steps
            costs[~at_goals] = makespan + 1

    return (
        all(terminated),
        makespan,
        np.sum(loses),
        np.sum(costs),
        np.mean(env.was_on_goal),
    )


def run_expert_algorithm(expert, env=None, observations=None, grid_config=None):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    makespan = 0
    costs = np.ones(env.get_num_agents())
    loses = np.ones(env.get_num_agents())

    expert.reset_states(env)

    while True:
        actions = expert.act(observations)

        original_pos = np.array([obs["global_xy"] for obs in observations])

        observations, rewards, terminated, truncated, infos = env.step(actions)

        new_pos = np.array([obs["global_xy"] for obs in observations])
        at_goals = np.array(env.was_on_goal)

        makespan += 1
        costs[~at_goals] = makespan + 1
        loses[~at_goals] += 1

        if all(terminated) or all(truncated):
            break

    return (
        all(terminated),
        makespan,
        np.sum(costs),
        np.sum(loses),
        np.mean(env.was_on_goal),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--test_name", type=str, default="in_distribution")
    parser.add_argument("--skip_n", type=int, default=None)
    parser.add_argument("--subsample_n", type=int, default=None)

    parser.add_argument("--set_expert_time_limit", type=int, default=None)

    parser.add_argument("--wandb_tag", type=str, default=None)

    args = parser.parse_args()
    print(args)

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(args)

    grid_configs = []

    if args.skip_n is not None:
        seeds = seeds[args.skip_n :]
    if args.subsample_n is not None:
        seeds = seeds[: args.subsample_n]

    for seed in seeds:
        grid_configs.append(_grid_config_generator(seed))

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    run_name = f"{args.test_name}_{args.expert_algorithm}"
    wandb.init(
        project=args.project_name,
        name=run_name,
        config=vars(args) | {"expert": True},
        entity=args.entity_name,
        tags=[args.wandb_tag] if args.wandb_tag is not None else None,
    )

    num_success = 0
    all_success, all_makespan = [], []
    all_sum_of_costs, all_partial_success_rates = [], []
    all_sum_of_loses = []
    num_samples = len(grid_configs)
    for i, grid_config in enumerate(grid_configs):
        print(f"Running expert on map {i + 1}/{num_samples}", end=" ")
        expert = expert_algorithm(inference_config)

        if args.set_expert_time_limit is not None:
            (
                success,
                makespan,
                sum_of_costs,
                sum_of_losses,
                partial_success_rate,
            ) = run_expert_algorithm_time_limit(
                expert,
                grid_config=grid_config,
                time_limit=args.set_expert_time_limit,
            )
        else:
            (
                success,
                makespan,
                sum_of_costs,
                sum_of_losses,
                partial_success_rate,
            ) = run_expert_algorithm(
                expert,
                grid_config=grid_config,
            )
        all_success.append(success)
        all_makespan.append(makespan)
        all_sum_of_costs.append(sum_of_costs)
        all_sum_of_loses.append(sum_of_losses)
        all_partial_success_rates.append(partial_success_rate)

        if success:
            num_success += 1

        success_rate = num_success / (i + 1)

        print(f"-- Success Rate: {success_rate}")
        wandb.log(
            {
                "success_rate": success_rate,
                "average_makespan": np.mean(all_makespan),
                "average_sum_of_costs": np.mean(all_sum_of_costs),
                "average_sum_of_loses": np.mean(all_sum_of_loses),
                "average_partial_success_rate": np.mean(all_partial_success_rates),
                "seed": grid_config.seed,
                "success": success,
                "makespan": makespan,
                "sum_of_costs": sum_of_costs,
                "sum_of_losses": sum_of_losses,
                "partial_success_rate": partial_success_rate,
            }
        )


if __name__ == "__main__":
    main()

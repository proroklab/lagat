import argparse
import pickle
import pathlib
import numpy as np
import torch

from pogema import pogema_v0

from magat_plus.run_expert import get_expert_dataset_file_name, add_expert_dataset_args
from magat_plus.dataset_loading import load_dataset

from grid_config_generator import grid_config_generator_factory

from magat_plus.additional_data.cost_to_go_calculator import (
    CostToGoCalculator,
    get_greedy_actions,
)

ADD_DATA_FILE_NAME_DEFAULTS = {
    "add_data_num_previous_actions": None,
    "add_data_cost_to_go": False,
    "add_data_greedy_action": False,
    "normalize_cost_to_go": True,
    "clamp_cost_to_go": None,
    "clamped_values_doubled": False,
}
ADD_DATA_FILE_NAME_KEYS = list(ADD_DATA_FILE_NAME_DEFAULTS.keys())
ADD_DATA_FILE_NAME_ALIASES = {
    "add_data_num_previous_actions": "nprev_acts",
    "add_data_cost_to_go": "c2g",
    "add_data_greedy_action": "greedy_act",
    "normalize_cost_to_go": "norm_c2g",
    "clamp_cost_to_go": "clamp_c2g",
    "clamped_values_doubled": "clamp_doubled",
}


def add_additional_data_args(parser):
    parser.add_argument("--add_data_num_previous_actions", type=int, default=None)
    parser.add_argument(
        "--add_data_cost_to_go", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--normalize_cost_to_go", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--clamp_cost_to_go", type=float, default=None)
    parser.add_argument(
        "--add_data_greedy_action", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--clamped_values_doubled", action=argparse.BooleanOptionalAction, default=False
    )

    return parser


def any_additional_data(args):
    additional_data = False
    idx = [None, None, None]
    cur_id = 0
    if args.add_data_cost_to_go:
        additional_data = True
        idx[0] = cur_id
        cur_id += 1
    if args.add_data_greedy_action:
        additional_data = True
        idx[1] = cur_id
        cur_id += 1
    if args.add_data_num_previous_actions is not None:
        additional_data = True
        idx[2] = (cur_id, args.add_data_num_previous_actions)
        cur_id += 1
    return additional_data, idx


def get_additional_data_file_name(args):
    dict_args = vars(args)

    file_name = get_expert_dataset_file_name(args)
    file_name = file_name[:-4]

    for key in sorted(ADD_DATA_FILE_NAME_KEYS):
        if dict_args[key] != ADD_DATA_FILE_NAME_DEFAULTS[key]:
            file_name += f"_{ADD_DATA_FILE_NAME_ALIASES[key]}_{dict_args[key]}"

    file_name = file_name + ".pkl"
    return file_name


def generate_additional_data(
    grid_config,
    all_actions,
    num_previous_actions=None,
    cost_to_go=False,
    normalized_cost_to_go=False,
    greedy_action=False,
    pad_cost_to_go=True,
    clamp_value=None,
    clamped_values_doubled=False,
    dtype="float32",
):
    if clamped_values_doubled:
        assert clamp_value is not None

    move_results = np.array(grid_config.MOVES)

    env = pogema_v0(grid_config)
    _, _ = env.reset()

    cost_to_go_calculator = None
    if cost_to_go or greedy_action:
        cost_to_go_calculator = CostToGoCalculator(
            env=env,
            obs_radius=grid_config.obs_radius,
            dtype=dtype,
            pad_cost_to_go=pad_cost_to_go,
            clamp_value=clamp_value,
            clamp_values_doubled=clamped_values_doubled,
        )

    all_additional_data = []
    if num_previous_actions is not None:
        prev_actions = np.zeros((env.num_agents, num_previous_actions), dtype=int)
    for actions in all_actions:
        additional_data = []
        if cost_to_go:
            cost_to_go_vals = cost_to_go_calculator.generate_cost_to_go(
                env=env, normalized=normalized_cost_to_go
            )
            additional_data.append(cost_to_go_vals)
        if greedy_action:
            if not cost_to_go:
                cost_to_go_vals = cost_to_go_calculator.generate_cost_to_go(
                    env=env, normalized=False
                )
            greedy_actions = get_greedy_actions(
                cost_to_go_vals, move_results, grid_config.obs_radius, dtype=dtype
            )
            additional_data.append(greedy_actions)
        if num_previous_actions is not None:
            additional_data.append(prev_actions)
            prev_actions = np.roll(prev_actions, shift=1, axis=1)
            prev_actions[:, 0] = actions

        additional_data = [torch.from_numpy(data) for data in additional_data]
        all_additional_data.append(additional_data)
        env.step(actions)
    return all_additional_data


def main():
    parser = argparse.ArgumentParser(description="Generate Additional Data")
    parser = add_expert_dataset_args(parser)
    parser = add_additional_data_args(parser)

    args = parser.parse_args()
    print(args)

    dataset = load_dataset(
        [get_expert_dataset_file_name],
        "raw_expert_predictions",
        args,
    )

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset
        seeds = seeds[seed_mask]
    elif not args.take_all_seeds:
        raise ValueError("Dataset is expected to have a seed_mask.")

    _grid_config_generator = grid_config_generator_factory(args)

    all_additional_data = []
    for sample_num, (seed, data) in enumerate(zip(seeds, dataset)):
        print(f"Generating Additional Data for map {sample_num + 1}/{args.num_samples}")
        grid_config = _grid_config_generator(seed)
        _, all_actions, _ = data

        additional_data = generate_additional_data(
            grid_config=grid_config,
            all_actions=all_actions,
            num_previous_actions=args.add_data_num_previous_actions,
            cost_to_go=args.add_data_cost_to_go,
            normalized_cost_to_go=args.normalize_cost_to_go,
            greedy_action=args.add_data_greedy_action,
            clamp_value=args.clamp_cost_to_go,
            clamped_values_doubled=args.clamped_values_doubled,
        )
        all_additional_data.extend(additional_data)

    file_name = get_additional_data_file_name(args)
    path = pathlib.Path(args.dataset_dir, "additional_data", file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_additional_data, f)


if __name__ == "__main__":
    main()

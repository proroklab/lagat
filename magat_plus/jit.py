import argparse
import pickle
import pathlib
import numpy as np
import wandb
from collections import OrderedDict
import time

import torch
from pogema import pogema_v0


from magat_plus.runtime_data_generation import get_runtime_data_generator
from magat_plus.run_expert import add_expert_dataset_args
from magat_plus.training_args import add_training_args
from magat_plus.convert_to_imitation_dataset import add_imitation_dataset_args
from grid_config_generator import grid_config_generator_factory

from magat_plus.generate_additional_data import add_additional_data_args
from magat_plus.magat.jit_agents import get_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="JIT compile models.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)

    parser.add_argument(
        "--test_in_distribution", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--get_validation_results", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--test_map_type", type=str, default="RandomGrid")
    parser.add_argument("--test_map_h", type=int, default=20)
    parser.add_argument("--test_map_w", type=int, default=20)
    parser.add_argument("--test_robot_density", type=float, default=0.025)
    parser.add_argument("--test_obstacle_density", type=float, default=0.1)
    parser.add_argument("--test_max_episode_steps", type=int, default=128)
    parser.add_argument("--test_obs_radius", type=int, default=3)
    parser.add_argument("--test_collision_system", type=str, default="soft")
    parser.add_argument("--test_on_target", type=str, default="nothing")

    parser.add_argument("--test_num_samples", type=int, default=2000)
    parser.add_argument("--test_dataset_seed", type=int, default=42)
    parser.add_argument("--test_dataset_dir", type=str, default="dataset")

    parser.add_argument("--test_comm_radius", type=int, default=7)
    parser.add_argument("--model_epoch_num", type=int, default=None)

    parser.add_argument("--test_name", type=str, default="in_distribution")
    parser.add_argument(
        "--test_wrt_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--test_min_dist", type=int, default=None)
    parser.add_argument("--test_max_dist", type=int, default=None)

    parser.add_argument("--test_map_types", type=str, default="random=0.1+maze=0.9")
    parser.add_argument("--test_map_w_min", type=int, default=16)
    parser.add_argument("--test_map_w_max", type=int, default=20)
    parser.add_argument("--test_num_agents", type=str, default="16+24+32")
    parser.add_argument("--test_obstacle_density_min", type=float, default=0.2)
    parser.add_argument("--test_obstacle_density_max", type=float, default=1.0)
    parser.add_argument("--test_go_straight_min", type=float, default=0.75)
    parser.add_argument("--test_go_straight_max", type=float, default=0.85)

    parser.add_argument("--test_wall_width_min", type=int, default=3)
    parser.add_argument("--test_wall_width_max", type=int, default=5)
    parser.add_argument("--test_wall_height_min", type=int, default=2)
    parser.add_argument("--test_wall_height_max", type=int, default=2)
    parser.add_argument("--test_side_pad", type=int, default=2)
    parser.add_argument("--test_horizontal_gap", type=int, default=1)
    parser.add_argument("--test_vertical_gap", type=int, default=3)
    parser.add_argument("--test_vertical_gap_min", type=int, default=None)
    parser.add_argument("--test_vertical_gap_max", type=int, default=None)
    parser.add_argument("--test_num_wall_rows_min", type=int, default=None)
    parser.add_argument("--test_num_wall_rows_max", type=int, default=None)
    parser.add_argument("--test_num_wall_cols_min", type=int, default=None)
    parser.add_argument("--test_num_wall_cols_max", type=int, default=None)
    parser.add_argument("--test_wfi_instance", action="store_true", default=False)
    parser.add_argument("--test_block_extra_space", action="store_true", default=True)

    parser.add_argument("--test_room_width_min", type=int, default=5)
    parser.add_argument("--test_room_width_max", type=int, default=9)
    parser.add_argument("--test_room_height_min", type=int, default=5)
    parser.add_argument("--test_room_height_max", type=int, default=9)
    parser.add_argument("--test_num_rows_min", type=int, default=3)
    parser.add_argument("--test_num_rows_max", type=int, default=5)
    parser.add_argument("--test_num_cols_min", type=int, default=3)
    parser.add_argument("--test_num_cols_max", type=int, default=5)
    parser.add_argument("--test_room_grid_uniform", action="store_true", default=True)
    parser.add_argument(
        "--test_room_only_centre_obstacles", action="store_true", default=False
    )

    parser.add_argument("--test_map_dir", type=str, default=None)
    parser.add_argument("--test_num_maps", type=int, default=1)

    parser.add_argument(
        "--test_ensure_grid_config_is_generatable",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--test_regulate_obstacle_density_max",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--skip_n", type=int, default=None)
    parser.add_argument("--subsample_n", type=int, default=None)

    parser.add_argument("--jit_model_storepath", type=str, default=None)

    args = parser.parse_args()

    print(args)

    assert args.jit_model_storepath is not None

    assert args.save_termination_state
    device = torch.device("cpu")

    rng = np.random.default_rng(args.test_dataset_seed)
    seeds = rng.integers(10**10, size=args.test_num_samples)

    _grid_config_generator = grid_config_generator_factory(args, testing=True)

    model, dataset_kwargs = get_model(args, device)

    num_parameters = count_parameters(model)
    print(f"Num Parameters: {num_parameters}")

    checkpoint_path = pathlib.Path(
        args.checkpoints_dir, f"epoch_{args.model_epoch_num}.pt"
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.eval()

    grid_config = _grid_config_generator(seeds[0])

    env = pogema_v0(grid_config=grid_config)
    observations, infos = env.reset()

    rt_data_generator = get_runtime_data_generator(
        grid_config=grid_config,
        args=args,
        dataset_kwargs=dataset_kwargs,
    )

    gdata = rt_data_generator(observations, env).to(device)

    with torch.jit.optimized_execution(True):
        opt_traced_model = torch.jit.trace(model, (gdata.x, dict(gdata)))

    save_path = pathlib.Path(args.jit_model_storepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    opt_traced_model.save(save_path)
    print(f"Saved JIT model to {args.jit_model_storepath}")


if __name__ == "__main__":
    main()

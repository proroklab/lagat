import argparse
import pickle
import pathlib
import numpy as np
import wandb

from pogema import GridConfig
import torch

from magat_plus.run_expert import add_expert_dataset_args
from magat_plus.magat.agents import get_model

from magat_plus.training_args import add_training_args
from magat_plus.convert_to_imitation_dataset import add_imitation_dataset_args

from magat_plus.run_model import run_model_on_grid
from grid_config_generator import grid_config_generator_factory

from magat_plus.generate_additional_data import add_additional_data_args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Test imitation learning model.")
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

    parser.add_argument("--svg_save_dir", type=str, default=None)
    parser.add_argument("--wandb_tag", type=str, default=None)

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    if args.test_in_distribution:
        num_agents = int(args.robot_density * args.map_h * args.map_w)

        train_id_max = int(
            args.num_samples * (1 - args.validation_fraction - args.test_fraction)
        )
        validation_id_max = train_id_max + int(
            args.num_samples * args.validation_fraction
        )

        rng = np.random.default_rng(args.dataset_seed)
        seeds = rng.integers(10**10, size=args.num_samples)
        if args.get_validation_results:
            seeds = seeds[train_id_max:validation_id_max]
        else:
            seeds = seeds[validation_id_max:]

        _grid_config_generator = grid_config_generator_factory(args)
    else:
        num_agents = int(args.test_robot_density * args.test_map_h * args.test_map_w)

        rng = np.random.default_rng(args.test_dataset_seed)
        seeds = rng.integers(10**10, size=args.test_num_samples)

        _grid_config_generator = grid_config_generator_factory(args, testing=True)

    model, dataset_kwargs = get_model(args, device)

    num_parameters = count_parameters(model)
    print(f"Num Parameters: {num_parameters}")

    if args.model_epoch_num is None:
        checkpoint_path = pathlib.Path(args.checkpoints_dir, "best.pt")
        if not checkpoint_path.exists():
            checkpoint_path = pathlib.Path(args.checkpoints_dir, "best_low_val.pt")
    else:
        checkpoint_path = pathlib.Path(
            args.checkpoints_dir, f"epoch_{args.model_epoch_num}.pt"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.eval()

    run_name = f"{args.test_name}_{args.run_name}"
    wandb.init(
        project=args.project_name,
        name=run_name,
        config=vars(args) | {"num_params": num_parameters},
        entity=args.entity_name,
        tags=[args.wandb_tag] if args.wandb_tag is not None else None,
    )

    if args.svg_save_dir is None:

        def aux_func(env, observations, actions, **kwargs):
            if actions is None:
                aux_func.original_pos = np.array(
                    [obs["global_xy"] for obs in observations]
                )
                aux_func.makespan = 0
                aux_func.costs = np.ones(env.get_num_agents())
            else:
                new_pos = np.array([obs["global_xy"] for obs in observations])
                at_goals = np.array(env.was_on_goal)
                aux_func.makespan += 1
                aux_func.original_pos = new_pos
                aux_func.costs[~at_goals] = aux_func.makespan + 1

    else:
        file_path = pathlib.Path(args.svg_save_dir)
        file_path.mkdir(parents=True, exist_ok=True)

        def aux_func(env, observations, actions, rtdg, **kwargs):
            if actions is None:
                aux_func.original_pos = np.array(
                    [obs["global_xy"] for obs in observations]
                )
                aux_func.makespan = 0
                aux_func.costs = np.ones(env.get_num_agents())
                aux_func.edge_index = []
            else:
                new_pos = np.array([obs["global_xy"] for obs in observations])
                at_goals = np.array(env.was_on_goal)
                aux_func.makespan += 1
                aux_func.original_pos = new_pos
                aux_func.costs[~at_goals] = aux_func.makespan + 1
            gdata = rtdg(observations, env)
            aux_func.edge_index.append(gdata.edge_index.detach().cpu().numpy())

    num_completed = 0
    num_tested = 0

    all_makespan = []
    all_partial_success_rate = []
    all_sum_of_costs = []

    if args.skip_n is not None:
        seeds = seeds[args.skip_n :]
    if args.subsample_n is not None:
        seeds = seeds[: args.subsample_n]

    for i, seed in enumerate(seeds):
        grid_config = _grid_config_generator(seed)
        success, env, _ = run_model_on_grid(
            model,
            device,
            grid_config,
            args,
            dataset_kwargs=dataset_kwargs,
            aux_func=aux_func,
            animation_monitor=args.svg_save_dir is not None,
        )
        makespan = aux_func.makespan
        costs = aux_func.costs

        num_tested += 1
        if success:
            num_completed += 1
        success_rate = num_completed / num_tested
        partial_success_rate = np.mean(env.was_on_goal)
        sum_of_costs = np.sum(costs)

        all_makespan.append(makespan)
        all_partial_success_rate.append(partial_success_rate)
        all_sum_of_costs.append(sum_of_costs)

        results = {
            "success_rate": success_rate,
            "average_makespan": np.mean(all_makespan),
            "average_partial_success_rate": np.mean(all_partial_success_rate),
            "average_sum_of_costs": np.mean(all_sum_of_costs),
            "seed": seed,
            "success": success,
            "makespan": makespan,
            "partial_success_rate": partial_success_rate,
            "sum_of_costs": sum_of_costs,
        }

        wandb.log(results)

        if args.svg_save_dir is not None:
            file_path = pathlib.Path(f"{args.svg_save_dir}", f"anim_{i}.svg")
            env.save_animation(file_path)

            file_path = pathlib.Path(f"{args.svg_save_dir}", f"edge_index_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(aux_func.edge_index, f)

        print(
            f"Testing Graph {i + 1}/{len(seeds)}, "
            f"Current Success Rate: {success_rate}"
        )


if __name__ == "__main__":
    main()

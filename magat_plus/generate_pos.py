import argparse
import pickle
import pathlib
import numpy as np
import torch

from magat_plus.run_expert import add_expert_dataset_args, get_expert_dataset_file_name
from magat_plus.dataset_loading import load_dataset
from magat_plus.convert_to_imitation_dataset import add_imitation_dataset_args


def get_pos_file_name(args):
    file_name = get_expert_dataset_file_name(args)
    file_name = file_name[:-4]

    if args.num_neighbour_cutoff is not None:
        file_name += f"_{args.num_neighbour_cutoff}_{args.neighbour_cutoff_method}_neighbour_cutoff"
    if args.use_edge_attr:
        file_name += "_pos"
    if len(file_name) > 0:
        file_name = file_name + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


def generate_graph_dataset(
    dataset,
    comm_radius,
    obs_radius,
    num_samples,
    save_termination_state,
    use_edge_attr=False,
    print_prefix="",
    id_offset=0,
    num_neighbour_cutoff=None,
    neighbour_cutoff_method=None,
    stack_with_np=True,
):
    dataset_agent_pos = []
    assert use_edge_attr

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])

            if use_edge_attr:
                dataset_agent_pos.append(global_xys)
    if not stack_with_np:
        return [torch.from_numpy(np.array(data)) for data in dataset_agent_pos]

    dataset_agent_pos = np.stack(dataset_agent_pos)
    return torch.from_numpy(dataset_agent_pos)


def main():
    parser = argparse.ArgumentParser(
        description="Convert to Imitation Learning Dataset"
    )
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)

    parser.add_argument("--use_edge_attr", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    dataset = load_dataset(
        [get_expert_dataset_file_name],
        "raw_expert_predictions",
        args,
    )
    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    graph_dataset = generate_graph_dataset(
        dataset,
        args.comm_radius,
        args.obs_radius,
        args.num_samples,
        args.save_termination_state,
        args.use_edge_attr,
        num_neighbour_cutoff=args.num_neighbour_cutoff,
        neighbour_cutoff_method=args.neighbour_cutoff_method,
        stack_with_np=not args.use_lists,
    )

    file_name = get_pos_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "positions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((graph_dataset), f)


if __name__ == "__main__":
    main()

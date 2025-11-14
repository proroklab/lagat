import numpy as np
import torch
from torch_geometric.data import Data

from magat_plus.convert_to_imitation_dataset import generate_graph_dataset
from magat_plus.imitation_dataset_pyg import MAPFGraphDataset

from magat_plus.additional_data.cost_to_go_calculator import (
    CostToGoCalculator,
    get_greedy_actions,
)
from magat_plus.generate_additional_data import any_additional_data


class BaseRuntimeDataGeneration:
    def __init__(self, **additional_kwargs):
        self.generators = dict()
        # Storing key separately to maintain order
        # (Could instead use OrderedDict instead)
        self.keys = []
        self.additional_kwargs = dict(**additional_kwargs)

    def register_datagenerator(self, key, generator):
        self.generators[key] = generator
        assert key not in self.keys
        self.keys.append(key)

    def register_params(self, key, value):
        self.additional_kwargs[key] = value

    def __call__(self, observations, env) -> Data:
        kwargs = {}
        for key in self.keys:
            kwargs[key] = self.generators[key](observations, env)
        return MAPFGraphDataset(**kwargs, **self.additional_kwargs)[0]


def get_graph_dataset_generator(
    comm_radius,
    obs_radius,
    num_neighbour_cutoff,
    neighbour_cutoff_method,
    distance_metric,
    random_edge_probs,
    dataset_kwargs,
):
    def _generator(observations, env):
        return generate_graph_dataset(
            dataset=[[[observations], [0], [0]]],
            comm_radius=comm_radius,
            obs_radius=obs_radius,
            num_samples=None,
            save_termination_state=True,
            use_edge_attr=dataset_kwargs["use_edge_attr"],
            print_prefix=None,
            num_neighbour_cutoff=num_neighbour_cutoff,
            neighbour_cutoff_method=neighbour_cutoff_method,
            distance_metric=distance_metric,
            random_edge_probs=random_edge_probs,
        )

    return _generator, "dense_dataset"


class AdditionalDataGenerator:
    def __init__(
        self,
        grid_config,
        num_previous_actions=None,
        cost_to_go=False,
        normalized_cost_to_go=False,
        greedy_action=False,
        clamp_value=None,
        clamped_values_doubled=False,
        dtype="float32",
    ):
        self.move_results = np.array(grid_config.MOVES)
        self.obs_radius = grid_config.obs_radius

        self.num_previous_actions = num_previous_actions
        self.cost_to_go = cost_to_go
        self.normalized_cost_to_go = normalized_cost_to_go
        self.greedy_action = greedy_action

        if num_previous_actions is not None:
            self.prev_actions = np.zeros(
                (grid_config.num_agents, num_previous_actions), dtype=int
            )
            self.prev_locations = None
            self.moves = np.array(grid_config.MOVES)
            self.moves = np.expand_dims(self.moves, axis=0)

        self.cost_to_go_calculator = None
        self.clamp_value = clamp_value
        self.clamped_values_doubled = clamped_values_doubled
        self.dtype = dtype

    def _generate_cost_to_go(self, env):
        if self.cost_to_go_calculator is None:
            self.cost_to_go_calculator = CostToGoCalculator(
                env=env,
                obs_radius=self.obs_radius,
                dtype=self.dtype,
                pad_cost_to_go=True,
                clamp_value=self.clamp_value,
                clamp_values_doubled=self.clamped_values_doubled,
            )
        return self.cost_to_go_calculator.generate_cost_to_go(
            env=env, normalized=self.normalized_cost_to_go
        )

    def __call__(self, observations, env):
        additional_data = []
        if self.cost_to_go:
            cost_to_go_vals = self._generate_cost_to_go(env)
            additional_data.append(cost_to_go_vals)
        if self.greedy_action:
            if not self.cost_to_go:
                cost_to_go_vals = self._generate_cost_to_go(env)
            greedy_actions = get_greedy_actions(
                cost_to_go_vals, self.move_results, self.obs_radius, dtype="float32"
            )
            additional_data.append(greedy_actions)
        if self.num_previous_actions is not None:
            new_locations = np.array(env.grid.get_agents_xy(ignore_borders=True))
            if self.prev_locations is None:
                actions = np.zeros(env.num_agents, dtype=int)
            else:
                diff = new_locations - self.prev_locations
                diff = np.expand_dims(diff, axis=1)
                actions = np.argmax(np.all(diff == self.moves, axis=-1), axis=-1)
            additional_data.append(self.prev_actions)
            self.prev_actions = np.roll(self.prev_actions, shift=1, axis=1)
            self.prev_actions[:, 0] = actions

        additional_data = [torch.from_numpy(data) for data in additional_data]
        return [additional_data]


def get_runtime_data_generator(
    grid_config,
    args,
    dataset_kwargs,
) -> BaseRuntimeDataGeneration:
    rt_data_generator = BaseRuntimeDataGeneration(**dataset_kwargs)

    generator, key = get_graph_dataset_generator(
        args.comm_radius,
        args.obs_radius,
        args.num_neighbour_cutoff,
        args.neighbour_cutoff_method,
        args.distance_metric,
        args.random_edge_probs,
        dataset_kwargs,
    )
    rt_data_generator.register_datagenerator(key, generator)

    load_additional_data, additional_data_idx = any_additional_data(args)
    if load_additional_data:
        key = "additional_data"
        generator = AdditionalDataGenerator(
            grid_config,
            num_previous_actions=args.add_data_num_previous_actions,
            cost_to_go=args.add_data_cost_to_go,
            normalized_cost_to_go=args.normalize_cost_to_go,
            greedy_action=args.add_data_greedy_action,
            clamp_value=args.clamp_cost_to_go,
            clamped_values_doubled=args.clamped_values_doubled,
        )

        rt_data_generator.register_datagenerator(key, generator)
        rt_data_generator.register_params("additional_data_idx", additional_data_idx)

    return rt_data_generator

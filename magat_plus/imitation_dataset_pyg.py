from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, scatter
from torch_geometric.data import Data


def convert_dense_graph_dataset_to_sparse_pyg_dataset(dense_dataset):
    new_graph_dataset = []
    (
        dataset_node_features,
        dataset_Adj,
        dataset_target_actions,
        dataset_terminated,
        graph_map_id,
    ) = dense_dataset
    for i in tqdm(range(dataset_node_features.shape[0])):
        edge_index, edge_weight = dense_to_sparse(dataset_Adj[i])
        new_graph_dataset.append(
            Data(
                x=dataset_node_features[i],
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=dataset_target_actions[i],
                terminated=dataset_terminated[i],
            )
        )
    return new_graph_dataset, graph_map_id


def decode_dense_dataset(dense_dataset, use_edge_attr):
    if use_edge_attr:
        return dense_dataset
    return *dense_dataset, None


def get_node_features(
    node_features,
    additional_data,
    additional_data_idx,
    index,
):
    # Updating to include cost-to-go data
    node_features = node_features[index]
    cost_to_go_idx = additional_data_idx[0]
    if cost_to_go_idx is None:
        return node_features
    cost_to_go = additional_data[index][cost_to_go_idx]
    cost_to_go = torch.unsqueeze(cost_to_go, dim=1)

    return torch.cat([node_features, cost_to_go], dim=1)


def add_additional_data(additional_data, additional_data_idx, index, dtype):
    kwargs = dict()
    if additional_data_idx[1] is not None:
        kwargs["greedy_action"] = additional_data[index][additional_data_idx[1]]
    if additional_data_idx[2] is not None:
        idx, _ = additional_data_idx[2]
        prev_actions = torch.nn.functional.one_hot(
            additional_data[index][idx], num_classes=5
        )
        prev_actions = prev_actions.reshape((prev_actions.shape[0], -1))
        prev_actions = prev_actions.to(dtype)
        kwargs["prev_actions"] = prev_actions
    return kwargs


class MAPFGraphDataset(Dataset):
    def __init__(
        self,
        dense_dataset,
        use_edge_attr,
        additional_data=None,
        additional_data_idx=[None, None, None],
        use_edge_attr_for_messages=None,
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr)
        self.use_edge_attr = use_edge_attr
        self.additional_data = additional_data
        self.additional_data_idx = additional_data_idx
        self.use_edge_attr_for_messages = use_edge_attr_for_messages

        if use_edge_attr_for_messages is not None:
            assert (
                self.use_edge_attr
            ), "Need to use edge_attr to use edge_attr_for_messages."

    def __len__(self) -> int:
        return len(self.dataset_node_features)

    def get_edge_index(self, index):
        return dense_to_sparse(self.dataset_Adj[index])

    def additional_kwargs(self, index, kwargs):
        return kwargs

    def return_data_item(self, kwargs):
        return Data(**kwargs)

    def __getitem__(self, index):
        edge_index, edge_weight = self.get_edge_index(index)
        edge_attr = None
        x = get_node_features(
            node_features=self.dataset_node_features,
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
        )
        y = self.dataset_target_actions[index]

        extra_kwargs = dict()
        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            pos_diff = agent_pos[edge_index[0]] - agent_pos[edge_index[1]]

            if self.use_edge_attr_for_messages is not None:
                if self.use_edge_attr_for_messages == "positions":
                    edge_attr = pos_diff.to(torch.float)
                elif self.use_edge_attr_for_messages == "dist":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
                elif self.use_edge_attr_for_messages == "manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                elif self.use_edge_attr_for_messages == "positions+dist":
                    edge_attr = pos_diff.to(torch.float)
                    dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                    edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
                elif self.use_edge_attr_for_messages == "positions+manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    manhattan = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                    edge_attr = torch.concatenate([edge_attr, manhattan], dim=-1)
                else:
                    raise ValueError(
                        f"Unsupported value for use_edge_attr_for_messages: {self.use_edge_attr_for_messages}."
                    )
            else:
                raise ValueError(
                    "use_edge_attr_for_messages must be set to use edge_attr."
                )

        extra_kwargs = extra_kwargs | add_additional_data(
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
            dtype=x.dtype,
        )
        kwargs = (
            dict(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                y=y,
                terminated=self.dataset_terminated[index],
            )
            | extra_kwargs
        )
        kwargs = self.additional_kwargs(index, kwargs)
        return self.return_data_item(kwargs)

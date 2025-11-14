import torch


class LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_function,
        accuracy_func,
        index_to_use=None,
        field_to_use="y",
        train_on_terminated_agents=True,
        extra_fields=[],
    ):
        super().__init__()
        self.loss_function = loss_function
        self.accuracy_func = accuracy_func
        self.index_to_use = index_to_use
        self.field_to_use = field_to_use
        self.train_on_terminated_agents = train_on_terminated_agents
        self.extra_fields = extra_fields

        self.accuracy_key = "accuracy"
        if field_to_use != "y":
            self.accuracy_key = f"{field_to_use}_accuracy"

    def get_accuracies(self, out, data, model, split="train"):
        if self.accuracy_func is None:
            return dict()

        if self.index_to_use is not None:
            out = out[self.index_to_use]
        target_actions = data[self.field_to_use]

        if not self.train_on_terminated_agents:
            out = out[~data.terminated]
            target_actions = target_actions[~data.terminated]

        acc = self.accuracy_func(out, target_actions)
        return {f"{split}_{self.accuracy_key}": acc}

    def forward(self, out, data, model):
        if self.index_to_use is not None:
            out = out[self.index_to_use]
        target_actions = data[self.field_to_use]

        if not self.train_on_terminated_agents:
            out = out[~data.terminated]
            target_actions = target_actions[~data.terminated]

        extra_fields = []
        for extra_field in self.extra_fields:
            ef = data[extra_field]
            if not self.train_on_terminated_agents:
                ef = ef[~data.terminated]
            extra_fields.append(ef)

        loss = self.loss_function(out, target_actions, *extra_fields)
        return loss


def default_acc(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, dim=-1) == y_true).detach().cpu()


def get_loss_function(args) -> torch.nn.Module:
    extra_fields = []
    loss_function = torch.nn.CrossEntropyLoss()
    acc_function = default_acc

    return LossWrapper(
        loss_function,
        accuracy_func=acc_function,
        train_on_terminated_agents=args.train_on_terminated_agents,
        extra_fields=extra_fields,
    )

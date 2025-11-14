from typing import Optional
import math
import torch.optim as optim


class BaseLRScheduler:
    def __init__(
        self,
        scheduler: optim.lr_scheduler._LRScheduler,
        step_on_batch: bool = False,
        step_on_epoch: bool = False,
        max_steps: Optional[int] = None,
    ):
        self.scheduler = scheduler
        self._step_on_batch = step_on_batch
        self._step_on_epoch = step_on_epoch
        self.cur_step = 0
        self.max_steps = max_steps

    def step_on_batch(self, *args, **kwargs):
        if self._step_on_batch:
            if self.max_steps is not None:
                if self.cur_step >= self.max_steps:
                    return
                self.cur_step += 1
            self.scheduler.step(*args, **kwargs)

    def step_on_epoch(self, *args, **kwargs):
        if self._step_on_epoch:
            self.scheduler.step(*args, **kwargs)


def get_estimated_total_number_of_steps(args, train_dataloader, fraction_of_oe=0.9):
    number_of_steps = len(train_dataloader) * args.num_epochs
    if args.skip_validation:
        return number_of_steps

    max_num_oe_runs = (
        args.num_epochs // args.validation_every_epochs
        - (args.run_oe_after // args.validation_every_epochs)
        - 1
    )
    max_num_oe_runs = max(max_num_oe_runs, 0)
    avg_num_oe_runs = max_num_oe_runs * 0.5
    num_oe_batches_per_run = (
        fraction_of_oe * args.num_run_oe + args.batch_size - 1
    ) // args.batch_size
    num_oe_steps = (
        avg_num_oe_runs * num_oe_batches_per_run * args.validation_every_epochs
    )
    num_oe_steps = math.ceil(num_oe_steps)

    return number_of_steps + num_oe_steps


def get_lr_scheduler(args, optimizer, train_dataloader):
    if args.lr_scheduler == "cosine-annealing":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr_end
        )
        return BaseLRScheduler(lr_scheduler, step_on_epoch=True)
    elif args.lr_scheduler == "one-cycle":
        number_of_steps = get_estimated_total_number_of_steps(args, train_dataloader)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr_start,
            total_steps=number_of_steps,
        )
        return BaseLRScheduler(
            lr_scheduler, step_on_batch=True, max_steps=number_of_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}.")

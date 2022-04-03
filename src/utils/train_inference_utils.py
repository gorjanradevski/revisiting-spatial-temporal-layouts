from typing import Dict

import torch
from torch import nn, optim


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int
):
    # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def add_weight_decay(model, weight_decay: float):
    # https://github.com/rwightman/pytorch-image-models/blob/48371a33b11fc30ca23ed8988619af08902b215b/timm/optim/optim_factory.py#L25
    decay = []
    no_decay = []
    skip_list = {}
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def move_batch_to_device(batch, device):
    return {
        key: val.to(device) if isinstance(val, torch.Tensor) else val
        for key, val in batch.items()
    }


class Criterion(nn.Module):
    def __init__(self, dataset_name: str):
        super(Criterion, self).__init__()
        self.loss_function = (
            nn.CrossEntropyLoss()
            if dataset_name == "something"
            else nn.BCEWithLogitsLoss()
        )

    def forward(self, logits: Dict[str, torch.Tensor], labels: torch.Tensor):
        return sum(
            [self.loss_function(logits[key], labels) for key in logits.keys()]
        ) / len(logits)

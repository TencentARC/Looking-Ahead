"""AdamW and step-addressed warmup/drop schedule."""

import torch


def build_generator_optimizer(config, generator):
    settings = config.optimizer_config.adamw
    return torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=float(settings.lr),
        betas=(float(settings.beta1), float(settings.beta2)),
        weight_decay=float(settings.weight_decay),
    )


class LinearLrWarmup:
    """Updates 1..N use 1/N..1; drop nodes address completed optimizer updates."""

    def __init__(self, optimizer, warmup_steps, drop_steps=(), drop_gamma=0.1):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.drop_steps = tuple(drop_steps)
        self.drop_gamma = float(drop_gamma)
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)

    def apply(self, step):
        if self.warmup_steps == 0 and not self.drop_steps:
            return 1.0
        update = step + 1
        factor = min(update / self.warmup_steps, 1.0) if self.warmup_steps else 1.0
        factor *= self.drop_gamma ** sum(update >= node for node in self.drop_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor
        return factor


def optimizer_learning_rate_metrics(optimizer):
    metrics = {"learning_rate": float(optimizer.param_groups[0]["lr"])}
    for index, group in enumerate(optimizer.param_groups[1:], start=1):
        metrics[f"learning_rate/group_{index}"] = float(group["lr"])
    return metrics

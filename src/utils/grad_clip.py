"""Disjoint main/auxiliary gradient clipping with root FSDP reductions."""

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.checkpoint_state import normalize_generator_fsdp_key as canonical_parameter_name

PREDICTIVE_PRIVATE_NAMESPACES = frozenset(
    {"predictive_multi_chunk_branches", "predictive_multi_chunk_shared_direct_modules"}
)


@dataclass(frozen=True)
class PredictiveGradClipPlan:
    """Canonical trainable-parameter names for disjoint gradient clipping."""

    main_names: frozenset[str]
    predictive_names: frozenset[str]


@dataclass(frozen=True)
class BoundPredictiveGradClipPlan:
    """Live parameters bound from a pre-FSDP predictive clipping plan."""

    main_parameters: tuple[nn.Parameter, ...]
    predictive_parameters: tuple[nn.Parameter, ...]


def build_predictive_grad_clip_plan(generator: nn.Module) -> PredictiveGradClipPlan:
    """Partition raw trainable generator parameters by aux-branch ownership."""

    main_names: set[str] = set()
    predictive_names: set[str] = set()
    for name, parameter in generator.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_name = canonical_parameter_name(name)
        destination = (
            predictive_names if is_predictive_private_parameter_name(canonical_name) else main_names
        )
        if canonical_name in main_names or canonical_name in predictive_names:
            raise ValueError(f"Duplicate trainable parameter name: {canonical_name}")
        destination.add(canonical_name)

    if not main_names:
        raise ValueError("Separate predictive gradient clipping found no main parameters.")
    if not predictive_names:
        raise ValueError(
            "separate_predictive_grad_clip=true requires trainable parameters under "
            f"one of {sorted(PREDICTIVE_PRIVATE_NAMESPACES)}."
        )
    return PredictiveGradClipPlan(
        main_names=frozenset(main_names),
        predictive_names=frozenset(predictive_names),
    )


def is_predictive_private_parameter_name(name: str) -> bool:
    """Return whether a canonical parameter belongs to either aux namespace."""
    return bool(PREDICTIVE_PRIVATE_NAMESPACES.intersection(name.split(".")))


def bind_predictive_grad_clip_plan(
    plan: PredictiveGradClipPlan,
    generator: nn.Module,
) -> BoundPredictiveGradClipPlan:
    """Bind canonical raw names to live parameters after FSDP wrapping."""

    expected_names = plan.main_names | plan.predictive_names
    bound: dict[str, nn.Parameter] = {}
    for name, parameter in generator.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_name = canonical_parameter_name(name)
        if canonical_name not in expected_names:
            raise ValueError(
                "FSDP-wrapped generator exposed an unexpected trainable parameter: "
                f"{name!r} -> {canonical_name!r}."
            )
        if canonical_name in bound:
            raise ValueError(f"Duplicate live trainable parameter name: {canonical_name}")
        bound[canonical_name] = parameter

    missing = sorted(expected_names - set(bound))
    if missing:
        raise ValueError(
            "Predictive gradient clipping parameters were not found after FSDP wrapping: "
            f"{missing[:10]}"
        )
    return BoundPredictiveGradClipPlan(
        main_parameters=tuple(bound[name] for name in sorted(plan.main_names)),
        predictive_parameters=tuple(bound[name] for name in sorted(plan.predictive_names)),
    )


def clip_predictive_grad_norms_(
    fsdp_generator: nn.Module,
    plan: BoundPredictiveGradClipPlan,
    *,
    max_main_norm: float,
    max_predictive_norm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip main and predictive-private gradients with separate FSDP collectives.

    The FSDP root owns the distributed norm calculation. Temporarily hiding the
    other disjoint group's gradients lets each call retain FSDP's FULL_SHARD,
    HYBRID_SHARD and NO_SHARD reduction semantics while applying an independent
    scale factor. Shared backbone parameters stay in the main group even when
    their gradients include a predictive-loss contribution.

    Returns:
        ``(total_norm, main_norm, predictive_norm)`` before clipping. The total
        norm preserves the historical global L2 ``grad_norm`` metric.
    """

    main_norm = _clip_parameter_group_with_fsdp_(
        fsdp_generator,
        selected_parameters=plan.main_parameters,
        hidden_parameters=plan.predictive_parameters,
        max_norm=max_main_norm,
    )
    predictive_norm = _clip_parameter_group_with_fsdp_(
        fsdp_generator,
        selected_parameters=plan.predictive_parameters,
        hidden_parameters=plan.main_parameters,
        max_norm=max_predictive_norm,
    )
    total_norm = torch.linalg.vector_norm(
        torch.stack(
            (
                main_norm.detach().to(dtype=torch.float32),
                predictive_norm.detach().to(dtype=torch.float32),
            )
        ),
        ord=2,
    )
    return total_norm, main_norm, predictive_norm


def _clip_parameter_group_with_fsdp_(
    fsdp_generator: nn.Module,
    *,
    selected_parameters: tuple[nn.Parameter, ...],
    hidden_parameters: tuple[nn.Parameter, ...],
    max_norm: float,
) -> torch.Tensor:
    if not selected_parameters:
        raise ValueError("Gradient clipping parameter group cannot be empty.")
    hidden_gradients = tuple(parameter.grad for parameter in hidden_parameters)
    try:
        for parameter in hidden_parameters:
            parameter.grad = None
        return fsdp_generator.clip_grad_norm_(max_norm)
    finally:
        for parameter, gradient in zip(
            hidden_parameters,
            hidden_gradients,
            strict=True,
        ):
            parameter.grad = gradient

"""Model-free path conventions for training and inference entry points."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def resolve_config_path(arg: str) -> Path:
    """Bare stems use configs/; relative config paths use the project root."""
    if not isinstance(arg, str) or not arg.strip():
        raise ValueError("config_path must be a nonempty path or config stem")
    path = Path(arg)
    if path.is_absolute():
        candidate = path
    elif "/" in arg or path.suffix.lower() in (".yaml", ".yml"):
        candidate = PROJECT_ROOT / path
    else:
        candidate = CONFIGS_DIR / f"{arg}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"config_path: file not found: {candidate}")
    return candidate.resolve()


def derive_expdir(config_path: Path) -> str:
    """Derive experiment paths using flat dash-split and nested config layouts."""
    path = config_path.resolve()
    try:
        relative = path.relative_to(CONFIGS_DIR)
    except ValueError:
        return str(PROJECT_ROOT / "outputs" / path.stem)
    if relative.parent == Path(".") and "-" in path.stem:
        method, setting = path.stem.split("-", 1)
        return str(PROJECT_ROOT / "outputs" / method / setting)
    return str(PROJECT_ROOT / "outputs" / relative.with_suffix(""))

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_path(value: str | None, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None:
        return None

    p = Path(value)
    if p.is_absolute():
        return p

    return base / p


def require_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    p = resolve_path(value, base=base)
    if p is None:
        raise ValueError("Required path is None")
    return p



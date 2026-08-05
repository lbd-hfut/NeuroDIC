"""Direct Python contract test for the compiled PIN model constructor."""

from pathlib import Path


def test_pin_model(project_root: Path) -> None:
    import torch
    from neurodic.models import make_pin_model

    model = make_pin_model(project_root / "config" / "pin_2d.yaml")
    coordinates = torch.zeros((4, 2), dtype=torch.float32)
    assert tuple(model.forward(coordinates).shape) == (4, 2)
    assert model.parameter_count() > 0


if __name__ == "__main__":
    test_pin_model(Path(__file__).resolve().parents[2])

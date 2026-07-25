import numpy as np
import torch

from literature_core.classifier import (
    build_mobilenet_classifier,
    patch_to_tensor,
    resolve_device,
)


def test_standard_mobilenet_adaptation_has_two_outputs_and_frozen_features() -> None:
    model = build_mobilenet_classifier(pretrained=False, freeze_backbone=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 224, 224)))
    assert output.shape == (2, 2)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())


def test_patch_preprocessing_is_chw_float() -> None:
    tensor = patch_to_tensor(np.zeros((224, 224, 3), dtype=np.uint8))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_numeric_device_matches_ultralytics_cli_style() -> None:
    assert str(resolve_device("0")) == "cuda:0"

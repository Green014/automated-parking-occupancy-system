import numpy as np
import torch

from literature_core.classifier import (
    CBAM,
    LeakyReLU6,
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


def test_paper_inspired_variant_supplements_se_and_changes_shallow_relu() -> None:
    model = build_mobilenet_classifier(
        pretrained=False,
        freeze_backbone=True,
        variant="cbam_leakyrelu6",
    )
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 224, 224)))
    assert output.shape == (1, 2)
    assert sum(isinstance(module, CBAM) for module in model.modules()) == 9
    assert any(isinstance(module, LeakyReLU6) for module in model.modules())
    cbam_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, CBAM)
        for parameter in module.parameters()
    ]
    assert cbam_parameters
    assert all(parameter.requires_grad for parameter in cbam_parameters)


def test_cbam_is_identity_at_initialization() -> None:
    module = CBAM(8).eval()
    inputs = torch.randn((2, 8, 7, 7))
    with torch.inference_mode():
        output = module(inputs)
    assert torch.allclose(output, inputs)


def test_leakyrelu6_caps_positive_values_and_keeps_negative_gradient() -> None:
    activation = LeakyReLU6(negative_slope=0.1)
    values = activation(torch.tensor([-2.0, 2.0, 8.0]))
    assert torch.allclose(values, torch.tensor([-0.2, 2.0, 6.0]))

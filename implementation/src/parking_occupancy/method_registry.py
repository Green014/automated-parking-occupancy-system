from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "baseline_methods.yaml"
)
REQUIRED_METHOD_IDS = frozenset({"B0", "B1", "E0", "T0"})
RUNNABLE_METHOD_IDS = frozenset({"B0", "B1", "T0"})


@dataclass(frozen=True, slots=True)
class RegisteredBaselineMethod:
    method_id: str
    canonical_name: str
    pipeline_experiment: str
    weights: str
    confidence: float
    image_size: int
    class_ids: tuple[int, ...]
    mapping_type: str
    minimum_slot_coverage: float | None
    data_role: str


def load_method_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate the closed B0/B1/E0/T0 naming registry."""

    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Baseline registry must use schema_version 1")
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("Baseline registry must contain a methods mapping")
    missing = REQUIRED_METHOD_IDS - set(methods)
    if missing:
        raise ValueError(
            f"Baseline registry is missing methods: {sorted(missing)}"
        )

    for method_id in REQUIRED_METHOD_IDS:
        method = methods[method_id]
        if not isinstance(method, dict):
            raise ValueError(f"{method_id} registry entry must be a mapping")
        detector = method.get("detector")
        mapping = method.get("mapping")
        if not isinstance(detector, dict) or not isinstance(mapping, dict):
            raise ValueError(f"{method_id} must declare detector and mapping")
        confidence = detector.get("confidence")
        image_size = detector.get("image_size")
        class_ids = detector.get("class_ids")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"{method_id} detector confidence is invalid")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"{method_id} detector image_size is invalid")
        if (
            not isinstance(class_ids, list)
            or not class_ids
            or any(not isinstance(value, int) for value in class_ids)
        ):
            raise ValueError(f"{method_id} detector class_ids are invalid")
        mapping_type = mapping.get("type")
        if mapping_type not in {
            "bbox_centre_inside_slot_polygon",
            "slot_polygon_coverage",
            "official_box_coverage",
        }:
            raise ValueError(f"{method_id} mapping type is invalid")
        threshold = mapping.get("minimum_slot_coverage")
        if threshold is not None and (
            not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1
        ):
            raise ValueError(f"{method_id} mapping threshold is invalid")
        if method_id in RUNNABLE_METHOD_IDS:
            if method.get("runnable") is not True:
                raise ValueError(f"{method_id} must be runnable")
            if method.get("pipeline_experiment") not in {"b0", "b1", "t0"}:
                raise ValueError(
                    f"{method_id} pipeline_experiment is invalid"
                )
    if methods["E0"].get("runnable") is not False:
        raise ValueError("Historical E0 must not be a runnable tuning method")
    return payload


def resolve_runnable_method(
    method_id: str,
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> RegisteredBaselineMethod:
    """Resolve one canonical runnable baseline without silent overrides."""

    normalized = method_id.upper()
    payload = load_method_registry(path)
    method = payload["methods"].get(normalized)
    if method is None:
        raise ValueError(f"Unknown registered method: {method_id}")
    if method.get("runnable") is not True:
        raise ValueError(
            f"{normalized} is historical-only; use its frozen artifacts/config"
        )
    detector = method["detector"]
    mapping = method["mapping"]
    return RegisteredBaselineMethod(
        method_id=normalized,
        canonical_name=str(method["canonical_name"]),
        pipeline_experiment=str(method["pipeline_experiment"]),
        weights=str(detector["weights"]),
        confidence=float(detector["confidence"]),
        image_size=int(detector["image_size"]),
        class_ids=tuple(int(value) for value in detector["class_ids"]),
        mapping_type=str(mapping["type"]),
        minimum_slot_coverage=(
            None
            if mapping["minimum_slot_coverage"] is None
            else float(mapping["minimum_slot_coverage"])
        ),
        data_role=str(method["data_role"]),
    )

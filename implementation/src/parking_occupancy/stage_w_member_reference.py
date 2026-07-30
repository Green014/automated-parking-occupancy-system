from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_registry import sha256_file
from .models import Detection, ParkingSlot
from .stage_v import (
    FrameOccupancyResult,
    SlotOccupancyState,
    validate_frame_result,
)


AUDITED_MEMBER_COMMIT = "12271576be39a4ac0eb456526eca122685799e8c"


class MemberReferenceUnavailable(RuntimeError):
    pass


class MemberReferenceBackend:
    """Adapter around the external member implementation.

    No member source, model, data, video, or template is copied by this class.
    The external checkout remains an explicit local dependency.
    """

    mode = "member-reference"

    def __init__(
        self,
        reference_root: Path,
        *,
        config_name: str = "config.json",
        require_audited_commit: bool = True,
    ) -> None:
        self.reference_root = reference_root.resolve()
        if not self.reference_root.is_dir():
            raise MemberReferenceUnavailable(
                "Member reference root is unavailable"
            )
        self.config_path = self.reference_root / config_name
        if not self.config_path.is_file():
            raise MemberReferenceUnavailable(
                "Member reference config is unavailable"
            )
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.commit = self._commit()
        if require_audited_commit and self.commit != AUDITED_MEMBER_COMMIT:
            raise MemberReferenceUnavailable(
                "Member reference checkout does not match the audited commit"
            )
        self._ParkingSpace: Any = None
        self._OccupancyManager: Any = None
        self._SlotClassifier: Any = None
        self._VehicleDetector: Any = None
        self._occupancy: Any = None
        self._inference: Any = None
        self._inference_kind = ""
        self._spaces: tuple[ParkingSlot, ...] = ()
        self._model_record: dict[str, Any] | None = None
        self.model_load_count = 0
        self._load_external_types()

    def _commit(self) -> str | None:
        if not (self.reference_root / ".git").exists():
            return None
        result = subprocess.run(
            ["git", "-C", str(self.reference_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _load_external_types(self) -> None:
        root = str(self.reference_root)
        sys.path.insert(0, root)
        try:
            parking_spaces = importlib.import_module("parking.parking_spaces")
            occupancy = importlib.import_module("parking.occupancy")
            slot_classifier = importlib.import_module("parking.slot_classifier")
            detector = importlib.import_module("parking.detector")
        except Exception as exc:
            raise MemberReferenceUnavailable(
                f"Member reference dependencies could not be imported: {exc}"
            ) from exc
        finally:
            if sys.path and sys.path[0] == root:
                sys.path.pop(0)
        self._ParkingSpace = parking_spaces.ParkingSpace
        self._OccupancyManager = occupancy.OccupancyManager
        self._SlotClassifier = slot_classifier.SlotOccupancyClassifier
        self._VehicleDetector = detector.VehicleDetector

    def prepare_slots(self, slots: Sequence[ParkingSlot]) -> None:
        slot_tuple = tuple(slots)
        if self._inference is not None:
            if [slot.slot_id for slot in slot_tuple] != [
                slot.slot_id for slot in self._spaces
            ]:
                raise MemberReferenceUnavailable(
                    "Member reference backend was already prepared for a "
                    "different slot map"
                )
            return
        member_spaces = [
            self._ParkingSpace(
                slot.slot_id,
                tuple(
                    (int(round(x)), int(round(y))) for x, y in slot.points
                ),
            )
            for slot in slot_tuple
        ]
        self._occupancy = self._OccupancyManager(
            member_spaces,
            self.config.get("occupancy"),
        )
        classifier = self.config.get("slot_classifier", {})
        if classifier.get("enabled", False):
            model_path = self.reference_root / str(
                classifier.get(
                    "model_path",
                    "models/slot_mobilenet_v3.pt",
                )
            )
            if not model_path.is_file():
                raise MemberReferenceUnavailable(
                    "Member reference classifier checkpoint is unavailable; "
                    "no model fallback was attempted"
                )
            self._inference = self._SlotClassifier(
                member_spaces,
                model_path,
                classifier.get("occupied_threshold", 0.75),
                classifier.get("input_size"),
            )
            self._inference_kind = "member_mobile_net_v3_slot_classifier"
            self._model_record = {
                "filename": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            }
        else:
            model_value = str(self.config.get("model", "yolov8n.pt"))
            model_path = self.reference_root / model_value
            if not model_path.is_file():
                raise MemberReferenceUnavailable(
                    "Member reference detector weights are unavailable; "
                    "automatic download and silent fallback are disabled"
                )
            self._inference = self._VehicleDetector(
                str(model_path),
                self.config.get("confidence", 0.25),
            )
            self._inference_kind = "member_yolo_bytetrack"
            self._model_record = {
                "filename": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            }
        self._spaces = slot_tuple
        self.model_load_count += 1

    def reset_state(self, slots: Sequence[ParkingSlot]) -> None:
        self.prepare_slots(slots)

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        frame_index: int,
        timestamp_s: float,
    ) -> FrameOccupancyResult:
        self.prepare_slots(slots)
        started = time.perf_counter()
        predictions: list[Mapping[str, Any]] = []
        if self._inference_kind == "member_yolo_bytetrack":
            vehicles = self._inference.track(frame)
        else:
            vehicles, predictions = self._inference.detect(frame)
        inference_end = time.perf_counter()
        self._occupancy.update(vehicles, timestamp_s)
        finished = time.perf_counter()
        score_by_slot = {
            str(row["slot_id"]): float(row["occupied_probability"])
            for row in predictions
        }
        states = []
        for slot in slots:
            runtime = self._occupancy.runtime[slot.slot_id]
            state_name = str(runtime.state.value)
            occupied = state_name in {"OCCUPIED", "VACANT_CANDIDATE"}
            states.append(
                SlotOccupancyState(
                    slot_id=slot.slot_id,
                    occupied=occupied,
                    evidence_score=score_by_slot.get(
                        slot.slot_id,
                        1.0 if occupied else 0.0,
                    ),
                    evidence_source=f"member_reference.{state_name.lower()}",
                    track_id=runtime.track_id,
                    details={"member_state": state_name},
                )
            )
        detections = tuple(
            Detection(
                bbox=tuple(float(value) for value in vehicle.bounding_box),
                confidence=float(vehicle.confidence),
                class_id=-1,
                class_name=str(vehicle.class_name),
                track_id=vehicle.track_id,
            )
            for vehicle in vehicles
        )
        result = FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=tuple(states),
            vehicle_detections=detections,
            timing_ms={
                "member_inference": (inference_end - started) * 1000.0,
                "member_occupancy": (finished - inference_end) * 1000.0,
                "backend_total": (finished - started) * 1000.0,
                "attributed_backend_total": (finished - started) * 1000.0,
            },
            warnings=(
                "Member-reference is an external optional comparison backend; "
                "its state logic is not the frozen C1/C2 method.",
            ),
        )
        validate_frame_result(result, slots)
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "method_id": "member-reference",
            "audited_commit": self.commit,
            "external_dependency": True,
            "direct_source_copy_in_adapter": False,
            "inference_kind": self._inference_kind or "not_loaded",
            "model": self._model_record,
            "model_load_count": self.model_load_count,
        }

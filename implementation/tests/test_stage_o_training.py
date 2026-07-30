from pathlib import Path

import cv2
import numpy as np
import yaml

from parking_occupancy.stage_n_lmot import LmotAnnotation
from parking_occupancy.stage_o_training import (
    annotations_to_yolo,
    build_combined_yolo_dataset,
    deterministic_frame_numbers,
)


def _image(path: Path, value: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(
        str(path), np.full((10, 20, 3), value, dtype=np.uint8)
    )


def _sequence(root: Path, sequence: str) -> None:
    sequence_root = root / sequence
    _image(sequence_root / "img_light_rgb" / "000001.jpg", 100)
    _image(sequence_root / "img_dark_rgb" / "000001.png", 10)
    (sequence_root / "gt").mkdir()
    (sequence_root / "gt" / "gt.txt").write_text(
        "1,1,1,1,10,5,1,3,1\n", encoding="utf-8"
    )
    (sequence_root / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence}\n"
        "imWidth=20\nimHeight=10\nseqLength=1210\n",
        encoding="utf-8",
    )


def test_deterministic_sampling_has_121_frames_and_fixed_endpoints() -> None:
    frames = deterministic_frame_numbers()

    assert len(frames) == 121
    assert frames[0] == 1
    assert frames[-1] == 1201
    assert all(right - left == 10 for left, right in zip(frames, frames[1:]))


def test_lmot_annotation_mapping_excludes_person_and_keeps_motor_classes() -> None:
    rows = [
        LmotAnnotation(1, 1, 0, 0, 10, 10, 1, 3, 1),
        LmotAnnotation(1, 2, 0, 0, 10, 10, 1, 1, 1),
        LmotAnnotation(1, 3, 5, 5, 10, 10, 1, 6, 1),
    ]

    labels = annotations_to_yolo(rows, width=20, height=10)

    assert len(labels) == 2
    assert all(row.startswith("0 ") for row in labels)
    assert labels[1].split()[4] == "0.50000000"


def test_combined_dataset_preserves_pair_groups_and_parking_mix(
    tmp_path: Path,
) -> None:
    lmot = tmp_path / "lmot"
    _sequence(lmot, "LMOT-train")
    _sequence(lmot, "LMOT-dev")
    ndispark = tmp_path / "ndispark"
    for index in range(112):
        name = f"{index:03d}.jpg"
        _image(ndispark / "images" / "train" / name)
        label = ndispark / "labels" / "train" / f"{index:03d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    output = tmp_path / "combined"

    result = build_combined_yolo_dataset(
        extracted_lmot_root=lmot,
        output_root=output,
        training_sequences=("LMOT-train",),
        development_sequences=("LMOT-dev",),
        ndispark_prepared_root=ndispark,
        frame_numbers=(1,),
    )

    assert result["cross_split_pair_count"] == 0
    assert result["pair_group_count"] == 2
    assert result["split_counts"]["train"]["lmot_images"] == 2
    assert result["split_counts"]["train"]["ndispark_images"] == 112
    assert result["split_counts"]["val"]["lmot_images"] == 2
    assert (output / "dataset.yaml").is_file()
    assert (output / "dataset_smoke.yaml").is_file()


def test_combined_dataset_can_declare_final_root_before_atomic_rename(
    tmp_path: Path,
) -> None:
    lmot = tmp_path / "lmot"
    _sequence(lmot, "LMOT-train")
    _sequence(lmot, "LMOT-dev")
    ndispark = tmp_path / "ndispark"
    for index in range(112):
        name = f"{index:03d}.jpg"
        _image(ndispark / "images" / "train" / name)
        label = ndispark / "labels" / "train" / f"{index:03d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    partial = tmp_path / "dataset.partial"
    final = tmp_path / "dataset"

    build_combined_yolo_dataset(
        extracted_lmot_root=lmot,
        output_root=partial,
        declared_root=final,
        training_sequences=("LMOT-train",),
        development_sequences=("LMOT-dev",),
        ndispark_prepared_root=ndispark,
        frame_numbers=(1,),
    )

    dataset = yaml.safe_load(
        (partial / "dataset.yaml").read_text(encoding="utf-8")
    )
    smoke_train = (
        partial / "smoke_train.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert dataset["path"] == str(final.resolve())
    assert smoke_train
    assert all(str(final.resolve()) in path for path in smoke_train)

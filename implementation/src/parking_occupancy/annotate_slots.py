from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .models import ParkingSlot
from .slots import SlotMap, save_slot_map


def _read_reference(path: Path, frame_index: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open image/video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def annotate(reference_path: Path, output_path: Path, frame_index: int = 0) -> None:
    frame = _read_reference(reference_path, frame_index)
    height, width = frame.shape[:2]
    current: list[tuple[float, float]] = []
    slots: list[ParkingSlot] = []
    window = "Parking slot annotation"

    def on_mouse(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _userdata: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    saved = False
    while True:
        canvas = frame.copy()
        for slot in slots:
            contour = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [contour], True, (0, 210, 0), 2, cv2.LINE_AA)
            center = tuple(
                np.mean(np.asarray(slot.points), axis=0).round().astype(int)
            )
            cv2.putText(
                canvas,
                slot.slot_id,
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 210, 0),
                1,
                cv2.LINE_AA,
            )

        if current:
            points = np.asarray(current, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [points],
                False,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
            for point in current:
                cv2.circle(
                    canvas,
                    (int(point[0]), int(point[1])),
                    4,
                    (0, 210, 255),
                    -1,
                )

        help_text = (
            "Left:add  Right:undo point  Enter:commit  "
            "Backspace:delete slot  S:save  Q:quit"
        )
        cv2.rectangle(canvas, (0, 0), (min(width, 900), 32), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            help_text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(current) >= 3:
            contour = np.asarray(current, dtype=np.float32).reshape((-1, 1, 2))
            if not cv2.isContourConvex(contour):
                print("Polygon rejected: points must form a convex slot.")
                continue
            slots.append(
                ParkingSlot(
                    slot_id=f"slot_{len(slots) + 1:03d}",
                    points=tuple(current),
                )
            )
            current.clear()
        elif key in (8, 127) and slots:
            slots.pop()
        elif key in (ord("s"), ord("S")):
            if not slots:
                print("Nothing saved: annotate at least one slot.")
                continue
            save_slot_map(
                SlotMap(
                    schema_version=1,
                    source_width=width,
                    source_height=height,
                    source=str(reference_path),
                    slots=tuple(slots),
                ),
                output_path,
            )
            saved = True
            print(f"Saved {len(slots)} slots to {output_path}")
            break
        elif key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow(window)
    if not saved:
        print("Annotation closed without saving.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate convex parking slots")
    parser.add_argument("--input", required=True, help="Reference image or video")
    parser.add_argument("--output", required=True, help="Output slot-map JSON")
    parser.add_argument("--frame", type=int, default=0, help="Video frame index")
    args = parser.parse_args()
    annotate(Path(args.input), Path(args.output), args.frame)


if __name__ == "__main__":
    main()


# Report-Ready Method and Contribution Text

These paragraphs are intentionally conservative and match the executed code.

## Method-source statement

The Part II system is a local integration of two literature-derived evidence
branches rather than an exact reproduction of one paper. First, manually
defined parking polygons are perspective-normalized with OpenCV and classified
by a standard torchvision MobileNetV3-Small adapted with a two-class head and
ImageNet transfer learning. The use of 224 x 224 slot patches and a lightweight
MobileNetV3 classifier is motivated by Yuldashev et al. [2], but this
implementation does not reproduce their LeakyReLU6, CBAM, or blueprint
separable-convolution modifications. Second, a pretrained YOLO-World model
[6] is prompted for vehicle categories; its object confidence is converted
into per-slot occupancy evidence using confidence-weighted polygon coverage.
YOLO-World does not detect vacant spaces directly. The two probabilities are
combined by an interpretable development-selected weighted sum, followed,
where continuous video is valid, by asymmetric EMA and ON/OFF hysteresis.
Every intermediate value and the final state are retained for audit.

## Contribution statement

The project contribution is an auditable parking-occupancy integration and
comparison: OpenCV slot normalization, an adapted MobileNetV3 branch,
YOLO-World vehicle evidence, explicit polygon mapping, probability-level
fusion, and corrected temporal evaluation are placed behind common
interfaces and evaluated against the preserved YOLOv8 baseline. It is not
claimed as a new detector, tracker, or reproduction of the modified
MobileNetV3 paper. The camera-holdout pilot found that YOLO-World alone
slightly improved macro F1 over the baseline, while development-selected
fusion did not generalize better; that negative result is retained as part of
the original holdout interpretation. A later three-camera rotation found that
fusion beat the baseline in two folds and tied it in one, but also showed that
the selected weights and thresholds varied by camera. A positive-only
external-domain check further showed that
hysteresis could greatly reduce flicker while simultaneously reducing
occupied recall, demonstrating why stability and correctness must be reported
separately. A subsequent three-sequence search reviewed 16 automated temporal
hypotheses and seven targeted apparent vacancies/departures, but none
described both a complete legal parking bay and a human-visible state change.
The project therefore retains the positive-only result and does not invent a
full E4 score from unsuitable regions.

## Continuous-data limitation statement

The available continuous Grand Bassin material is useful for testing
positive-state stability but not for a complete occupied/vacant transition
evaluation. Motion-based preannotations primarily proposed access-road
vehicles, image-edge fragments, stationary parked vehicles with detector
dropout, and queued vehicles overlapping adjacent bays. Apparent empty
regions were visually identified as circulation lanes, no-parking hatching,
or occupied/occluded spaces. Consequently, vacant recall, false-occupied
rate, transition latency, mixed-class flicker, IDF1, and HOTA are left
unclaimed until suitable manual ground truth is collected.

## Structural comparison

```text
Existing baseline
frame -> YOLOv8 -> optional ByteTrack -> polygon mapping
      -> confidence-aware EMA/hysteresis -> slot state

Literature-core workflow
frame + slot polygons
  |-> OpenCV warp -> adapted MobileNetV3 -> P_cls ---------|
  |-> YOLO-World -> confidence/coverage mapping -> P_det --|-> weighted P_occ
                                                            -> EMA/hysteresis
                                                            -> final slot state
```

The baseline stays operational and unchanged. The second workflow is
independent under `implementation/literature_core/`.

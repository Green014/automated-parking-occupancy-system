# Report-Ready Method and Contribution Text

These paragraphs are intentionally conservative and match the executed code.

## Method-source statement

The Part II system is a local integration of two literature-derived evidence
branches rather than an exact reproduction of one paper. First, manually
defined parking polygons are perspective-normalized with OpenCV and classified
by a standard torchvision MobileNetV3-Small adapted with a two-class head and
ImageNet transfer learning. The use of 224 x 224 slot patches and a lightweight
MobileNetV3 classifier is motivated by Yuldashev et al. [2], but this
E1a is the standard adapted model. E1b is explicitly paper-inspired rather
than an exact reproduction: it preserves the pretrained SE path and adds an
identity-initialized CBAM supplement; shallow LeakyReLU6 is ablated
separately, while BSConv is not claimed. Second, a pretrained YOLO-World
model [6] is prompted for vehicle categories; its object confidence is
converted into per-slot evidence using confidence multiplied by slot
coverage. This quantity is not a native occupancy probability, and
YOLO-World does not detect vacant spaces directly. E3a retains the historical
raw weighted sum. The proposed E3b first fits separate monotonic calibrators
to the classifier score and detector evidence, then applies a non-negative
logistic fusion to the calibrated log-odds. All E3b fitting and threshold
selection uses complete development cameras rather than randomly divided
slots. Every intermediate value and final state is retained for audit.

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

## Experiment statement

The 27 selected PKLot images are treated only as camera-grouped method
development data. In leave-one-camera-out development diagnostics, E3b
improved branch Brier score and ECE but reached a camera-equal Macro F1 of
0.983959, below E3a's 0.991353; this negative result was retained. The final
configuration was then frozen before downloading predictions from the
official ODbL-1.0 CNR-EXT external holdout. Evaluation covered 4,081 complete
images and 144,965 slot labels, with 95% confidence intervals obtained by
resampling complete images. E0 achieved the best external Macro F1
(0.966766, 95% CI 0.965044-0.968463), followed by E2 (0.963589). E1b improved
over E1a (0.910801 versus 0.894361), but neither E3a (0.921187) nor E3b
(0.909022) surpassed the detector baselines. E3b produced the lowest
false-occupied rate (0.004263) but a high false-free rate (0.162813), so it is
not presented as the most accurate method.

## Calibration limitation statement

Calibration quality on PKLot development did not guarantee threshold
transfer across datasets. On CNR-EXT, calibrating YOLO-World evidence reduced
Brier score from 0.187201 to 0.059467, but calibrating the classifier worsened
Brier score from 0.074689 to 0.099345 and ECE from 0.044488 to 0.119239. E3b
was better calibrated than E3a yet had lower Macro F1 because the frozen
decision threshold favored vacant recall over occupied recall. No threshold,
coefficient, checkpoint, or prompt was changed after the external result was
viewed.

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
  |-> OpenCV warp -> E1a/E1b MobileNetV3 -> score -----------|
  |-> YOLO-World -> confidence x coverage evidence ----------|
                                                             |-> E3a raw weighted
                                                             |-> E3b calibrate
                                                                 -> non-negative
                                                                    logistic fusion
```

The baseline stays operational and unchanged. The second workflow is
independent under `implementation/literature_core/`.

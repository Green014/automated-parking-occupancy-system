# Third-Party Notices

This repository contains project-authored integration code and depends on
third-party software. Dependency source code and model weights are not bundled
unless explicitly identified.

## Runtime dependencies

| Component | Role | License / source |
|---|---|---|
| Ultralytics | YOLOv8 inference, ByteTrack, and TrackTrack integration | AGPL-3.0; <https://github.com/ultralytics/ultralytics> |
| TrackTrack | Optional multi-object tracking method exposed through Ultralytics | MIT author repository; <https://github.com/kamkyu94/TrackTrack> |
| TrackEval | Optional MOT metric implementation | MIT; <https://github.com/JonathonLuiten/TrackEval> |
| OpenCV | Image and video processing | <https://github.com/opencv/opencv> |
| PyTorch and torchvision | E1b classifier inference and training | <https://github.com/pytorch/pytorch> |

Users are responsible for complying with the applicable dependency licenses,
including the Ultralytics licensing terms, when redistributing or deploying a
derived system.

## Datasets and weights

Downloaded datasets, source images, extracted frames, and trained model
checkpoints are not redistributed in this repository. Dataset names, hashes,
and result summaries are retained only where needed to document experiment
provenance.

## Project code

No license is granted for the project-authored code at present. Public
visibility permits inspection through GitHub but does not by itself grant
permission to redistribute or reuse the code outside applicable law and the
GitHub Terms of Service.

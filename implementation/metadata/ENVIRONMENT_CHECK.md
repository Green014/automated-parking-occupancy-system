# Environment Check

Checked: 24 July 2026, Asia/Shanghai

## Result

| Component | Observed value | Implementation consequence |
|---|---|---|
| Python | 3.12.13 bundled runtime | A project-local virtual environment was created at `.venv/` |
| OpenCV | 4.13.0 | Installed and used for decoding, polygons, rendering, and MP4 output |
| PyTorch | 2.13.0+cu130 | CUDA build installed in the project environment |
| Ultralytics | 8.4.104 | Pretrained YOLOv8 and ByteTrack adapter available |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | CUDA inference is feasible |
| VRAM | 6,144 MiB total; 5,435 MiB free during check | Start with YOLOv8n, batch size 1, 640 px |
| NVIDIA driver | 610.62 | Recent driver is available |
| Driver CUDA capability | CUDA UMD 13.3 | A compatible PyTorch CUDA wheel can use the driver |
| CUDA Toolkit / `nvcc` | Not installed | Not required for ordinary PyTorch/Ultralytics inference |
| Storage | C: 1,862.06 GB total; 143.58 GB free | Small samples and selected datasets are feasible; avoid 44-724 GB sources |

`nvidia-smi` was available and reported compute capability 8.6. The initial
Python environment had NumPy but did not have `cv2`, `torch`, `ultralytics`,
`shapely`, `yaml`, or `pytest`.

After setup, `torch.cuda.is_available()` returned `True`,
`torch.version.cuda` returned `13.0`, and PyTorch identified the GPU as
`NVIDIA GeForce RTX 3060 Laptop GPU`. The implemented test suite passes
14 tests.

The project does not require Shapely for the first version. Polygon processing
is implemented with OpenCV so that the coursework's OpenCV requirement is
visible in the core pipeline.

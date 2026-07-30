# Third-Party Notices

This repository is a public research/course project, not a commercial or
deployment-grade product. The project source is distributed under
AGPL-3.0-only; third-party software, datasets, pretrained initialization, and
generated model artifacts remain subject to their own terms. Inclusion here
is attribution, not a claim that third-party materials are relicensed.

No third-party dataset, pretrained model, local vendor checkout, or generated
runtime output is included in the public source candidate.

## Runtime and development dependencies

| Component | License boundary | Project use | Authoritative notice |
|---|---|---|---|
| Ultralytics | AGPL-3.0 or a separately obtained Enterprise license | YOLO training/runtime framework used by the historical D1 workflow | <https://docs.ultralytics.com/> |
| PyTorch | BSD-style license, with bundled components under their respective notices | Tensor runtime and training framework | <https://github.com/pytorch/pytorch/blob/main/LICENSE> |
| torchvision | BSD-3-Clause | MobileNetV3-Small architecture and ImageNet-pretrained initialization used by E1b | <https://github.com/pytorch/vision/blob/main/LICENSE> |
| OpenCV | Apache-2.0 for OpenCV 4.5.0 and later | Image/video processing and the Classic teaching baseline | <https://opencv.org/license/> |
| Flask | BSD-3-Clause | Optional local Stage W dashboard server | <https://github.com/pallets/flask> |
| PyYAML | MIT | YAML configuration and release metadata | <https://github.com/yaml/pyyaml> |
| NumPy | BSD-3-Clause base license; wheels can contain separately licensed bundled components | Numerical operations | <https://github.com/numpy/numpy> |
| Shapely | BSD-3-Clause; its GEOS dependency is LGPL-2.1 | Geometry operations | <https://github.com/shapely/shapely> |
| Matplotlib | PSF-derived, BSD-compatible Matplotlib license | Research figures | <https://matplotlib.org/stable/project/license.html> |
| lap | BSD-2-Clause | Optional linear-assignment dependency | <https://pypi.org/project/lap/> |
| TrackEval | MIT | Optional historical tracking evaluation | <https://github.com/JonathonLuiten/TrackEval> |

The exact installed environment can contain transitive packages not listed
above. Redistributors are responsible for preserving the notices shipped with
those packages.

## Dataset attribution and model provenance

The public source candidate contains no training or evaluation images.

### NDISPark

D1 was fine-tuned on NDISPark. The frozen source audit made on 2026-07-27
recorded Zenodo record 6560823 as Open Data Commons Attribution License 1.0
(ODC-By-1.0). Preserve dataset attribution and database notices when using or
redistributing derived databases.

- Dataset record: <https://zenodo.org/records/6560823>
- ODC-By 1.0 terms: <https://opendatacommons.org/licenses/by/>

### PKLot

E1b was trained on PKLot parking-space patches. The official project page
publishes PKLot under Creative Commons Attribution 4.0 International
(CC-BY-4.0) and requests citation of the dataset paper.

- Dataset page and citation: <https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/>
- CC BY 4.0 terms: <https://creativecommons.org/licenses/by/4.0/>

## Adapted dashboard interface

The Stage W HTML/CSS presentation is adapted from:

- repository: <https://github.com/prestzy/OpenCV-Car-Parking>
- audited commit: `12271576be39a4ac0eb456526eca122685799e8c`

The upstream code owner authorized course integration and public
redistribution of the adapted interface. The project owner records that
authorization anonymously; the underlying evidence is retained privately
outside this repository. See
`implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md` for the anonymous
public record and the code-boundary audit.

## Excluded or unresolved material

Local vendor directories, datasets or assets without a confirmed public
license are excluded. The source manifest also excludes private permission
evidence, member-supplied models/data/videos, model weights, generated outputs,
virtual environments, and caches. No license is inferred for an excluded
asset merely because it exists in a local worktree.

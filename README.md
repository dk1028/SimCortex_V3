<p align="center">
  <img src="docs/assets/simcortex-logo.png" alt="SimCortex logo" width="320"/>
</p>

# SimCortex v2.0 with Fixed-Initialization MRI-Only Deformation

SimCortex v2.0 is a modular and reproducible framework for cortical surface reconstruction in **MNI152 space**.

This branch additionally contains the **fixed-initialization MRI-only deformation experiments**, in which the original subject-specific deformation initialization is replaced by a fixed cortical template.

The official SimCortex v2.0 pipeline remains available in this repository. The fixed-initialization experiments modify only the input contract and architecture of the **deformation stage**.

> **Previous version:** The original ShapeMI/MICCAI 2025 conference implementation is preserved as:
>
> * [SimCortex v1.0.0 release](https://github.com/Neuro-iX/SimCortex/releases/tag/v1.0.0)
> * [Legacy v1 branch](https://github.com/Neuro-iX/SimCortex/tree/legacy/v1-shapemi2025)
> * [Conference paper](https://arxiv.org/abs/2507.06955)

---

## SimCortex pipeline

The official SimCortex v2.0 workflow contains four stages:

1. **Preprocessing**

   * register MRI and FreeSurfer-derived resources to MNI152

2. **Segmentation**

   * predict a 9-class cortical segmentation using a 3D U-Net

3. **Initial Surfaces (InitSurf)**

   * generate subject-specific white and pial surfaces
   * generate SDF and cortical ribbon probability resources

4. **Deformation**

   * deform the initial cortical surfaces toward target anatomy

The official pipeline is:

```text
T1 MRI
  |
  v
Preprocessing
  |
  v
9-class segmentation
  |
  v
InitSurf
  | \
  |  +--> ribbon probability
  |
  +-----> subject-specific white/pial surfaces
                |
                v
        deformation network
        MRI + ribbon probability
                |
                v
         final cortical surfaces
```

---

# Fixed-Initialization MRI-Only Variant

The fixed-initialization experiments investigate whether the original subject-specific InitSurf initialization can be replaced by a **single fixed cortical template**.

The deformation-stage contract becomes:

```text
MNI152 T1 MRI
        +
fixed initial cortical surfaces
        |
        v
MRI-only deformation network
        |
        v
four multi-scale SVFs
        |
        v
Gaussian smoothing
        |
        v
SVF scaling-and-squaring integration
        |
        v
displacement interpolation at mesh vertices
        |
        v
vertex updates
        |
        v
final cortical surfaces
```

The important differences from the official deformation stage are:

| Component                    | Official SimCortex v2.0   | Fixed-initialization variant |
| ---------------------------- | ------------------------- | ---------------------------- |
| MRI input                    | yes                       | yes                          |
| Ribbon probability           | yes                       | no                           |
| Initial surfaces             | subject-specific InitSurf | fixed cortical template      |
| Deformation channels         | `c_in=2`                  | `c_in=1`                     |
| Encoder                      | `DualMUNetV2`             | `MUNetV2`                    |
| Geometry/probability encoder | yes                       | no                           |
| `GeomInject` fusion          | yes                       | no                           |
| Multi-scale SVFs             | 4                         | 4                            |
| Gaussian SVF smoothing       | yes                       | yes                          |
| SVF integration              | yes                       | yes                          |
| Vertex updates               | yes                       | yes                          |

The fixed-initialization experiment therefore changes **initialization and conditioning**, while preserving the core deformation mechanics.

Importantly, segmentation and InitSurf have **not been removed from SimCortex itself**. Their implementations remain available as upstream pipeline stages. They are simply not required to provide subject-specific initialization to the final fixed-template deformation experiment.

---

## Fixed-Initialization Experimental Arms

Five final initialization conditions are included:

### Sphere

A canonical subdivision-7 icosphere initialization.

```text
Sphere
```

Each surface contains:

```text
163842 vertices
327680 faces
```

---

### Random

An unsmoothed subject-specific initialization selected reproducibly from the OASIS1 training set.

```text
subject: sub-0447
dataset: OASIS1
seed: 2025
```

---

### Original150k

A fixed anatomical initialization derived from HCP subject with the smallest ASSD value:

```text
sub-298051
```

The subject was selected using white-surface ASSD-based medoid selection.

The surfaces were then processed using ordinary Taubin smoothing:

```text
iterations = 150000
lambda = 0.5
nu = 0.5
```

---

### Curv0

The same anatomical source lineage is processed with:

```text
collision-aware Taubin smoothing
        |
        v
largest-connected-component cleanup
        |
        v
Geometry Central adjustEdgeLengths()
        |
        v
target edge length = 0.618 mm
curvatureAdaptation = 0
```

---

### Curv1

Curv1 uses the same collision-aware and LCC-cleaned lineage as Curv0, but uses curvature-adaptive remeshing:

```text
target edge length = 0.618 mm
curvatureAdaptation = 1.0
```

The five arms use the **same MRI-only deformation architecture**.

They are therefore five different **initialization conditions**, not five different neural-network architectures.

Detailed template provenance is available in:

```text
docs/reproducibility/FIXED_TEMPLATES.md
```

---

## Reproducibility Documentation

Detailed experimental provenance is provided under:

```text
docs/reproducibility/
```

Important documents include:

* [Reproducibility index](docs/reproducibility/README.md)
* [Official SimCortex vs fixed-initialization variant](docs/reproducibility/UPSTREAM_DELTA.md)
* [Baseline and experiment provenance](docs/reproducibility/BASELINE.md)
* [Fixed-template provenance](docs/reproducibility/FIXED_TEMPLATES.md)
* [Checkpoint compatibility](docs/reproducibility/CHECKPOINT_COMPATIBILITY.md)
* [Real-MRI inference reproduction](docs/reproducibility/INFERENCE_REPRODUCTION.md)

Machine-readable provenance is stored under:

```text
manifests/reproducibility/
```

This includes:

```text
source_archives.tsv
training_configs.tsv
checkpoints.tsv
fixed_templates.tsv
checkpoint_compatibility.tsv
real_mri_smoke.tsv
```

---

## Reproducibility Status

The reconstructed fixed-initialization experiment currently has the following verification status:

```text
Official upstream baseline                     PASS
Source reconstruction                          PASS
Configuration reconstruction                   PASS
Five-arm experiment reconstruction             PASS
Fixed-template provenance                      PASS
Exact checkpoint provenance                    PASS
Strict checkpoint compatibility                PASS
Historical inference launcher provenance       CAPTURED
Clean Narval real-MRI functional reproduction  PASS
Exact Neuro-Ix sample40 replay                 NOT PERFORMED
```

All five final historical deformation checkpoints were verified to load strictly into the clean MRI-only implementation with:

```text
missing keys = 0
unexpected keys = 0
shape mismatches = 0
```

A functional real-MRI inference test was also completed using a historical HCP_YA test MRI:

```text
subject = sub-135932
space = MNI152
```

The test completed successfully and generated all four cortical surfaces with preserved mesh topology.

The clean functional reproduction should **not** be interpreted as an exact replay of the historical Neuro-Ix sample40 external evaluation.

---

# Installation

From the repository root:

```bash
python -m pip install -e .
```

Verify the CLI:

```bash
simcortex --help
simcortex fs-to-mni --help
simcortex seg --help
simcortex initsurf --help
simcortex deform --help
```

## Stage-specific extras

```bash
# Stage 1: preprocessing
python -m pip install -e ".[preproc]"

# Stage 2: segmentation
python -m pip install -e ".[seg]"

# PyTorch runtime
python -m pip install -e ".[torch]"

# Optional deformation metrics and collision evaluation
python -m pip install -e ".[deform-metrics]"
```

To install all declared extras:

```bash
python -m pip install -e ".[preproc,seg,torch,deform-metrics]"
```

---

## PyTorch3D

The deformation stack requires PyTorch3D.

PyTorch3D is not declared as a generic pip extra because installation must match the selected PyTorch and CUDA versions.

Install a compatible PyTorch3D build separately or use the validated container environment.

---

# Configuration

Hydra configuration files are provided under:

```text
src/simcortex/configs/
  seg/
  initsurf/
  deform/
```

Fixed-initialization experiment overlays are stored under:

```text
configs/fixed_init/
```

The five final overlays are:

```text
sphere.yaml
random.yaml
original150k.yaml
curv0.yaml
curv1.yaml
```

The fixed-initialization deformation configuration uses:

```yaml
dataset:
  use_probability_map: false
  use_fixed_initial_surface: true
  fixed_template_root: null

model:
  c_in: 1
  strict_load: true
  sigma: 1
  n_steps: 8
  inshape: [184, 224, 184]
  c_hid: [8, 16, 32, 64, 128, 128]
  gn_groups: 8
  dropout: 0.1
```

A valid fixed-template directory must contain:

```text
lh_pial_smoothed.ply
lh_white_smoothed.ply
rh_pial_smoothed.ply
rh_white_smoothed.ply
```

---

# Data and Folder Conventions

A typical dataset layout is:

```text
datasets/<dataset-name>/
  bids/
  derivatives/
    freesurfer-7.4.1/
    sc-preproc/
    sc-seg/
    sc-initsurf/
    sc-deform/
  splits/
    <dataset>_split.csv
```

Typical naming conventions:

* subjects: `sub-XXXX`
* sessions: `ses-01`
* MNI space: `space-MNI152`
* segmentation: `desc-seg9_dseg`
* deformation outputs: `desc-deform`

---

# Split File Format

## Single-dataset split

Minimum columns:

```text
subject
split
```

Example:

```csv
subject,split
sub-0001,train
sub-0002,val
sub-0003,test
```

## Multi-dataset split

Use:

```text
subject
split
dataset
```

Example:

```csv
subject,split,dataset
sub-100307,test,HCP_YA
sub-101915,test,HCP_YA
sub-0001,test,OASIS1
```

The dataset names in the CSV must match the corresponding Hydra dataset-root keys.

---

# Recommended Workflows

## Official SimCortex workflow

```text
Preprocessing
     |
     v
Segmentation
     |
     v
InitSurf
     |
     v
Deformation
```

In command form:

1. create `sc-preproc`
2. train/run segmentation to create `sc-seg`
3. generate `sc-initsurf`
4. train/infer/evaluate deformation to create `sc-deform`

---

## Fixed-Initialization MRI-Only workflow

For the final fixed-template experiments:

```text
preprocessed MNI152 MRI
        +
fixed cortical template
        |
        v
MRI-only deformation
        |
        v
final cortical surfaces
```

The final deformation stage therefore does not require a subject-specific InitSurf directory or ribbon probability map.

The fixed template is provided through:

```text
dataset.fixed_template_root
```

---

# Stage 1 - Preprocessing

Stage 1 creates MNI152-aligned MRI and associated FreeSurfer-derived resources.

It can:

1. export FreeSurfer MRI volumes
2. apply optional N4 correction
3. estimate native-to-MNI registration
4. resample volumes into MNI152
5. transform cortical surfaces into MNI152 space

Typical command:

```bash
simcortex fs-to-mni \
  --freesurfer-root /path/to/datasets/<dataset>/derivatives/freesurfer-7.4.1 \
  --out-deriv-root /path/to/datasets/<dataset>/derivatives/sc-preproc \
  --mni-template /path/to/SimCortex/src/MNI152_T1_1mm.nii.gz \
  --transform-type affine \
  --n4 \
  --with-aparc-aseg \
  --with-filled \
  -v
```

A typical output includes:

```text
sc-preproc/
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
      surfaces/
        sub-XXXX_ses-01_space-MNI152_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_pial.surf.ply
```

The fixed-initialization MRI-only deformation experiments still require the MNI152 T1 MRI produced by preprocessing.

---

# Stage 2 - Segmentation

The official SimCortex segmentation stage trains or applies a 3D U-Net to predict a 9-class segmentation in MNI152 space.

Typical output:

```text
sc-seg/
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz
```

Example:

```bash
simcortex seg train \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.root=/path/to/simcortex-runs/seg/exp01
```

Inference example:

```bash
simcortex seg infer \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/sc-seg
```

This stage remains part of official SimCortex.

It is not required for the **final fixed-template deformation input path** when an existing MNI152 MRI and fixed template are supplied.

---

# Stage 3 - Initial Surfaces (InitSurf)

The official InitSurf stage generates subject-specific cortical initialization from saved segmentation predictions.

Outputs include:

```text
sc-initsurf/
  sub-XXXX/
    ses-01/
      anat/
        ..._desc-lh_white_sdf.nii.gz
        ..._desc-rh_white_sdf.nii.gz
        ..._desc-lh_pial_sdf.nii.gz
        ..._desc-rh_pial_sdf.nii.gz
        ..._desc-ribbon_sdf.nii.gz
        ..._desc-ribbon_prob.nii.gz

      surfaces/
        ..._hemi-L_white.surf.ply
        ..._hemi-L_pial.surf.ply
        ..._hemi-R_white.surf.ply
        ..._hemi-R_pial.surf.ply
```

Example:

```bash
simcortex initsurf generate \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.seg_root=/path/to/datasets/<dataset>/derivatives/sc-seg \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/sc-initsurf
```

In official SimCortex, these subject-specific surfaces and ribbon resources are passed to deformation.

In the fixed-initialization experiments, this subject-specific initialization dependency is replaced by a fixed template.

---

# Stage 4 - Deformation

## Official SimCortex deformation

The official deformation stage receives:

```text
MNI152 MRI
+
ribbon probability
+
subject-specific InitSurf surfaces
```

The default official architecture uses:

```text
DualMUNetV2
c_in = 2
```

with separate MRI and geometry/probability branches.

---

## Fixed-Initialization MRI-Only Deformation

The fixed-init deformation stage instead receives:

```text
MNI152 MRI
+
fixed template surfaces
```

with:

```text
MUNetV2
c_in = 1
```

Ribbon probability input is disabled:

```yaml
dataset.use_probability_map: false
```

and fixed initialization is enabled:

```yaml
dataset.use_fixed_initial_surface: true
```

A fixed template must be supplied:

```yaml
dataset.fixed_template_root: /path/to/fixed/template
```

---

## Deformation Mechanism

The network predicts four stationary velocity fields:

```text
SVF1
SVF2
SVF3
SVF4
```

Each deformation update follows:

```text
predicted SVF
     |
     v
Gaussian smoothing
     |
     v
scaling-and-squaring integration
     |
     v
displacement field
     |
     v
interpolation at mesh vertices
     |
     v
vertex update
```

The four updates are applied sequentially to the cortical mesh.

---

## Fixed-Initialization Inference Example

A fixed-template inference run requires:

```text
MNI152 preprocessing root
split file
fixed template
MRI-only checkpoint
output root
```

Representative configuration:

```yaml
dataset:
  split_name: test
  use_probability_map: false
  use_fixed_initial_surface: true
  fixed_template_root: /path/to/template

model:
  ckpt_path: /path/to/deform_best_rmse.pth
  c_in: 1
  strict_load: true
  n_steps: 8
```

---

# Offline Fixed-Template Preparation

Several geometry operations used in the experimental templates are **template-preparation operations**, not per-subject prediction postprocessing.

These include:

```text
Taubin smoothing
collision-aware smoothing
largest-connected-component cleanup
Geometry Central adjustEdgeLengths()
uniform edge-only remeshing
curvature-adaptive remeshing
```

The resulting mesh becomes the shared initialization supplied to training and inference.

These operations are therefore performed **before** subject-level deformation inference.

---

# Evaluation

Deformation evaluation can compute surface geometry and collision metrics.

Typical outputs include:

```text
surface_metrics.xlsx
collision_metrics.xlsx
collision_metrics_enhanced.xlsx
collision_summary.xlsx
```

The historical final external evaluation also included exact FCL collision evaluation.

The five fixed-initialization conditions were compared using the same evaluation framework so that changes could be attributed primarily to initialization geometry.

---

# Docker

Docker support is provided as an execution environment for SimCortex.

The main image supports:

* Stage 1: preprocessing
* Stage 2: segmentation
* Stage 3: InitSurf
* Stage 4: deformation

Basic test:

```bash
docker run --rm simcortex:2.0.0 simcortex --help
```

GPU test:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

For complete Docker usage see:

```text
docker/README.md
```

Docker Hub:

[kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)

---

# License

See the repository `LICENSE` file.

# Fixed-Initialization MRI-Only Cortical Surface Deformation

This repository contains the fixed-initialization MRI-only cortical surface deformation experiments derived from the deformation stage of SimCortex v2.0.

The project studies whether a shared cortical template can replace subject-specific InitSurf initialization while preserving the same downstream deformation mechanism.

## Pipeline

```text
preprocessed MNI152 T1 MRI
        +
fixed cortical template
        |
        v
MRI-only deformation network
        |
        v
four multi-scale stationary velocity fields
        |
        v
Gaussian smoothing
        |
        v
scaling-and-squaring integration
        |
        v
displacement interpolation at mesh vertices
        |
        v
vertex updates
        |
        v
four final cortical surfaces
        |
        v
surface and collision evaluation
```

The deformation model uses one MRI input channel:

```text
model.c_in = 1
dataset.use_probability_map = false
dataset.use_fixed_initial_surface = true
```

Segmentation and subject-specific InitSurf are not part of this repository's runtime pipeline.

## Fixed-Initialization Conditions

Five initialization conditions are provided:

| Arm | Initialization |
|---|---|
| **Sphere** | subdivision-7 icosphere |
| **Random** | unsmoothed OASIS1 `sub-0447`, selected with seed 2025 |
| **Original150k** | HCP_YA `sub-298051` anatomical template after 150,000 Taubin smoothing iterations |
| **Curv0** | collision-aware 150k template with LCC cleanup and edge-only remeshing, `curvatureAdaptation=0` |
| **Curv1** | same lineage as Curv0 with curvature-adaptive remeshing, `curvatureAdaptation=1.0` |

All five arms use the same MRI-only deformation architecture. The experimental variable is the fixed initialization geometry.

The final training overlays are stored in:

```text
configs/fixed_init/
  sphere.yaml
  random.yaml
  original150k.yaml
  curv0.yaml
  curv1.yaml
```

## Repository Structure

```text
configs/
  fixed_init/              final five experiment overlays

src/simcortex/
  cli/                     deformation-only command line interface
  configs/deform/          base train, inference, and evaluation configs
  deform/
    data/                   fixed-template MRI dataloader
    models/                 MRI-only deformation model
    utils/                  coordinate transforms
    train.py
    inference.py
    eval.py
  utils/
    collision_backend.py    FCL collision backend

resources/
  README.md                 required template layout

docs/reproducibility/       template/checkpoint and execution provenance
manifests/reproducibility/  machine-readable SHA256 manifests
tests/                      fixed-initialization contract tests
```

## Requirements

Python 3.10 or newer is recommended.

Install the package:

```bash
python -m pip install -e ".[torch]"
```

For collision-aware evaluation:

```bash
python -m pip install -e ".[torch,deform-metrics]"
```

The deformation stack also requires a PyTorch3D build compatible with the installed PyTorch and CUDA versions.

## Data Requirements

This project starts from **preprocessed MNI152 T1 MRI**.

Training and evaluation additionally require MNI152 target cortical surfaces. A split CSV is required with:

```text
subject,split,dataset
```

for multi-dataset experiments, or:

```text
subject,split
```

for a single dataset.

Restricted HCP_YA and OASIS1 MRI data are not included in this repository.

## Fixed Templates

Set the resource root:

```bash
export SIMCORTEX_RESOURCES_ROOT=/path/to/resources
export SIMCORTEX_RUNS_ROOT=/path/to/runs
```

Each fixed-template directory must contain:

```text
lh_pial_smoothed.ply
lh_white_smoothed.ply
rh_pial_smoothed.ply
rh_white_smoothed.ply
```

The expected directory names and exact SHA256 identities are documented in:

```text
resources/README.md
manifests/reproducibility/fixed_templates.tsv
docs/reproducibility/FIXED_TEMPLATES.md
```

## Training

The CLI is deformation-only.

Example for Sphere:

```bash
simcortex-fixedinit train \
  --torchrun \
  --nproc-per-node 2 \
  user_config=configs/fixed_init/sphere.yaml \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/hcpya/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/oasis1/derivatives/sc-preproc
```

Replace `sphere.yaml` with one of:

```text
random.yaml
original150k.yaml
curv0.yaml
curv1.yaml
```

for the other initialization conditions.

## Inference

Inference requires:

- preprocessed MNI152 T1 MRI
- a fixed four-surface template
- an MRI-only deformation checkpoint
- a subject split file

Example:

```bash
simcortex-fixedinit infer \
  dataset.path=/path/to/dataset/derivatives/sc-preproc \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  dataset.fixed_template_root=/path/to/fixed_template \
  model.ckpt_path=/path/to/deform_best_rmse.pth \
  outputs.out_root=/path/to/predictions
```

The output for each subject contains:

```text
lh white
lh pial
rh white
rh pial
```

deformed cortical surfaces in MNI152 space.

## Evaluation

Example:

```bash
simcortex-fixedinit eval \
  dataset.path=/path/to/dataset/derivatives/sc-preproc \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  outputs.pred_root=/path/to/predictions \
  outputs.out_dir=/path/to/evaluation
```

Evaluation includes surface-distance metrics and optional collision metrics when FCL is available.

## Reproducibility

The repository records:

- the five final experiment configurations
- exact fixed-template SHA256 hashes
- exact final checkpoint SHA256 hashes
- strict checkpoint compatibility results
- a successful clean real-MRI inference smoke test

See:

- `docs/reproducibility/FIXED_TEMPLATES.md`
- `docs/reproducibility/CHECKPOINT_COMPATIBILITY.md`
- `docs/reproducibility/INFERENCE_REPRODUCTION.md`
- `manifests/reproducibility/`

The real-MRI smoke test is a functional reproduction and is not claimed to be an exact replay of the historical Neuro-Ix sample40 evaluation.

## Upstream

This project is derived from the deformation framework of:

**SimCortex v2.0**
https://github.com/Neuro-iX/SimCortex

The pinned upstream baseline used for this work is:

```text
6f3cb21af9763807190407e9725331b0f2e78bed
```

Original SimCortex authorship and licensing are preserved in the repository history and `LICENSE`.

## License

See `LICENSE`.

# Fixed-initialization training overlays

These files reproduce the five final MRI-only fixed-initialization
training arms on top of:

`src/simcortex/configs/deform/train.yaml`

The overlays are intended for the base configuration introduced at
commit:

`56dfc560e6f9b7c32656d0cf48c0127c8c885d0a`

## Input contract

All five final arms use:

- MRI only
- `dataset.use_probability_map=false`
- `dataset.use_fixed_initial_surface=true`
- `model.c_in=1`
- batch size 1 per GPU
- 600 epochs
- seed 2025

The effective optimization batch is 12 for every arm.

Sphere and Random use:

- 2 GPUs
- gradient accumulation 6

Original150k, Curv0, and Curv1 use:

- 3 GPUs
- gradient accumulation 4

## Portable paths

Set these environment variables before training:

`SIMCORTEX_RESOURCES_ROOT`

Root containing the fixed-template directories.

`SIMCORTEX_RUNS_ROOT`

Root where deformation training runs are written.

Dataset locations and the train/validation split are intentionally not
stored here. Supply them through the base configuration or Hydra command
line overrides.

Restricted HCP/OASIS MRI data are not part of this repository.

## Historical provenance

`manifests/reproducibility/training_configs.tsv` records the SHA256 of
each recovered historical final-arm training configuration together with
its final experiment name, template identity, GPU count, and effective
batch.

Historical Sphere and Random launchers explicitly invoked their training
configuration through `user_config=...`.

The clean overlays for all five arms use the same supported
`user_config` merge mechanism. For Original150k, Curv0, and Curv1, the
overlays represent the verified recovered final training configurations;
they do not claim that the exact historical launcher syntax was recovered.

# Real-MRI Inference Reproduction

## Scope

This document records a functional real-MRI inference reproduction of the fixed-initialization MRI-only SimCortex experiment.

The clean reconstruction is derived from the official SimCortex repository:

- Official repository: `https://github.com/Neuro-iX/SimCortex`
- Official pinned baseline: `6f3cb21af9763807190407e9725331b0f2e78bed`
- Clean reconstruction repository: `https://github.com/dk1028/SimCortex_V3`
- Clean reconstruction commit tested: `911f3105c117b2e5872e1c9a79abe79be2bd1728`

This test is a functional real-data reproduction. It is not an exact replay of the historical Neuro-Ix sample40 external evaluation.

## Reconstructed inference contract

The tested deformation path used:

- preprocessed MNI152 T1 MRI
- one MRI input channel (`model.c_in=1`)
- no ribbon probability-map input
- a fixed initial cortical-surface template
- no subject-specific InitSurf requirement
- strict checkpoint loading
- eight SVF integration steps

The tested clean configuration had:

```text
dataset.use_probability_map=false
dataset.use_fixed_initial_surface=true
model.c_in=1
model.strict_load=true
model.n_steps=8
model.inshape=[184,224,184]
```

This differs from the official upstream SimCortex deformation configuration, which uses subject-specific initialization resources and a two-channel MRI-plus-ribbon-probability deformation input.

## Historical real-MRI input

The real MRI was selected from the same historical dataset split and preprocessing roots used by the five final fixed-initialization training arms.

Historical split:

```text
/project/ctb-sbouix/kavehets/datasets/splits/dataset_split.csv
```

Split SHA256:

```text
68f42ef3ed09373685de5d1dd144c7d3d72b7f8083850057dcccd5780756188d
```

The split contains 515 subjects across HCP_YA and OASIS1.

Test subject:

```text
dataset = HCP_YA
subject = sub-135932
session = ses-01
split = test
```

MRI:

```text
/project/ctb-sbouix/kavehets/datasets/hcpya-u100/derivatives/sc-preproc-0.2/sub-135932/ses-01/anat/sub-135932_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
```

MRI SHA256:

```text
6e44da590a495235b9546c55bab770a451cdbf98f9fb870d3d443295f3fd24fb
```

The stored MRI shape was `182 x 218 x 182`. The clean deformation dataloader center-padded/cropped the MRI to the configured model shape `184 x 224 x 184`.

## Historical final Sphere checkpoint

Checkpoint:

```text
/project/ctb-sbouix/kimdowoo/results/simcortex/deform/exp37_hcpya+oasis1_onlyMRI_icosphere_clean_2gpu/checkpoints/deform_best_rmse.pth
```

Size:

```text
21311114 bytes
```

SHA256:

```text
555f198379db1aa5f585aec586a939ffbf2680fbf3d86c06fb0979f6a701e8a1
```

The checkpoint was loaded with `strict=True`.

## Fixed Sphere template

The tested fixed template was:

```text
/project/ctb-sbouix/kimdowoo/workspace/SimCortex/resources/v2c_icosphere_fixed_template
```

Surface SHA256 values:

```text
lh_pial
7366a007877024070327019473cb87b04ac496a42d95db315551dc5a498e283a

lh_white
11e0e500ef15a2a350dda0718285ea9fe6052ce760a2a6dbf12e44d67a6c7528

rh_pial
0271ecd291c5cd93306c0d9b64787597523f063ea71ee1d41691de4527e054fe

rh_white
d09d8ca8cfbda0ddb5900d30dd962649dbc522963d1ff1dc37a73ed3b8216da8
```

Each Sphere surface contains:

```text
163842 vertices
327680 faces
```

## Runtime

The test used the historical Narval Python environment:

```text
Python: /scratch/kimdowoo/simcortex_env_py310_ready/bin/python
PyTorch: 2.1.0
NumPy: 1.24.3
Pandas: 2.0.3
Nibabel: 5.2.0
Trimesh: 4.1.3
Hydra: 1.3.2
OmegaConf: 2.3.0
```

The successful historical Apptainer runtime pattern was reused with `apptainer/1.3.5` and the CUDA 12.1.1 / cuDNN 8 runtime container.

## Slurm execution

Functional reproduction job:

```text
JobID: 2016129
Job name: sph_repro_smoke
GPU: 1 x A100
State: COMPLETED
ExitCode: 0:0
Elapsed: 00:02:53
```

The test produced exactly four final cortical surfaces.

## Mesh-integrity result

All four output surfaces passed the following checks:

- finite output vertices
- vertex-count preservation
- face-count preservation
- exact face-array/topology preservation
- non-empty output geometry

Results:

| Surface | Vertices | Faces | Mean displacement (mm) | RMSE displacement (mm) | Max displacement (mm) | Valid |
|---|---:|---:|---:|---:|---:|---|
| LH pial | 163842 | 327680 | 23.410794299441 | 27.216540992311 | 80.315456918114 | PASS |
| LH white | 163842 | 327680 | 21.741105293407 | 25.428591961840 | 78.781457684285 | PASS |
| RH pial | 163842 | 327680 | 24.673414630604 | 28.553631246982 | 78.383925364928 | PASS |
| RH white | 163842 | 327680 | 22.999078444198 | 26.662185837083 | 78.047847485936 | PASS |

Final integrity assertion:

```text
FOUR_OUTPUT_SURFACES_VALID=True
```

The displacement statistics above measure movement relative to the fixed Sphere initialization. They are not surface-accuracy metrics and should not be interpreted as ASSD or prediction error against ground truth.

## Output-surface hashes

The four generated cortical surfaces were independently verified on the Mac after transfer of the frozen Narval evidence archive.

```text
LH pial
7aa21f962d79f654fdd8eadc5b831283d50d1ad62f8a534252ead9035fb7f1ca

LH white
265472f44f25c9b95b3a1e05d7e68c985bb3bd6624cc7ec5d4fc2af8404a580b

RH pial
289c002d168a0cec7c30f4bad93953abd9a5bf187dc57abdcff6334c4b1bc82e

RH white
742bf2be414ef00398a834cddf21ff04a83b0c070cf54760c73fb032a945118e
```

The Mac-side four-surface SHA256 manifest has SHA256:

```text
d6b3837805881552efd119d31fa60cc4504a928bb6146bf8ecc1e86021796b73
```

## Preserved evidence

The functional reproduction evidence was frozen on Narval and transferred to the Mac.

Final evidence archive:

```text
simcortex_real_mri_smoke_911f310_job2016129.tar.gz
```

Archive SHA256:

```text
a53d66e88c90bdf27304cde906e69db546d8ccaaab41979e141ad489aede96cb
```

The transferred archive passed gzip integrity verification and retained the same SHA256 on the Mac.

The internal artifact manifest contains 18 files and has SHA256:

```text
f5a25c4d2b4b880dd7631556528592d35dd90531c1629c70f7aea738e06a4b4c
```

All 18 manifested files passed SHA256 verification after extraction on the Mac.

Additional evidence hashes:

```text
Slurm scontrol snapshot
e2ec363eaeda13fe50f84c13927ee6370b7a3c32b823522838848fb18c9a08ef

Mesh-integrity report
e7bd956c39819dcd07a372d88401e12f258cd4268ad25f092cac9afadd739244
```

## Interpretation

This result establishes that the clean MRI-only fixed-initialization source reconstruction can load an exact historical final checkpoint and fixed template and perform end-to-end deformation inference on a real historical HCP_YA MNI152 MRI.

The executed functional path was:

```text
historical MNI152 T1 MRI
+
fixed Sphere cortical surfaces
        |
        v
MRI-only deformation network
        |
        v
multi-scale stationary velocity fields
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
four final cortical surfaces
```

The result strengthens the earlier strict checkpoint compatibility audit from static state-dict compatibility to functional real-data execution.

## Reproducibility status

The resulting evidence hierarchy is:

```text
Official upstream baseline                     PASS
Source reconstruction                          PASS
Configuration reconstruction                   PASS
Fixed-template provenance                      PASS
Exact checkpoint provenance                    PASS
Strict checkpoint compatibility                PASS
Historical inference launcher provenance       CAPTURED
Clean Narval real-MRI functional reproduction  PASS
Exact Neuro-Ix sample40 replay                 NOT PERFORMED
```

## Limitation

This test must not be described as an exact replay of the historical Neuro-Ix sample40 external evaluation.

The historical Neuro-Ix inference launchers and path provenance were preserved, but the corresponding Neuro-Ix MRI/preprocessing payload was not captured in the final local Neuro-Ix snapshot before access ended.

Therefore:

```text
Historical Neuro-Ix inference provenance: CAPTURED
Clean Narval real-MRI functional reproduction: PASS
Exact Neuro-Ix sample40 replay: NOT PERFORMED
```

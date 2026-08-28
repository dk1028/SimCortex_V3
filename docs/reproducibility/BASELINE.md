# Fixed-Initialization Reproducibility Baseline

## Upstream baseline

This reproducibility branch is based on the following upstream SimCortex revision:

* Upstream repository: `Neuro-iX/SimCortex`
* Upstream branch: `main`
* Pinned upstream commit: `6f3cb21af9763807190407e9725331b0f2e78bed`
* Reproducibility branch: `repro/fixed-initialization`

The fixed-initialization implementation is maintained as an additive extension of this pinned upstream baseline.

## Final deformation input contract

The final fixed-initialization experiments use:

* one T1-weighted MRI input channel (`c_in = 1`);
* fixed initial cortical surfaces;
* no probability-map input to the deformation network.

Conceptually, the final deformation pipeline is:

```text
T1-weighted MRI
        +
fixed initial cortical surfaces
        |
        v
MRI-only deformation network
        |
        v
final cortical surfaces
```

The historical development workspace contains probability-map functionality and probability-map resources. These are retained as historical implementation artifacts, but they are not part of the final fixed-initialization deformation input contract.

## Historical execution environments

The original experiments were executed across two systems. This distinction is preserved because the two systems served different roles in the final experimental workflow.

### Narval

Narval was used for:

* template subject selection;
* fixed-template construction;
* spherical template generation;
* ordinary Taubin smoothing;
* collision-aware smoothing;
* largest-connected-component cleanup;
* Geometry Central remeshing;
* neural-network training.

### Neuro-Ix

Neuro-Ix was used for:

* final model inference;
* external multi-dataset evaluation;
* scannerRAS surface conversion;
* exact `python-fcl` collision evaluation.

The historical workflow should therefore not be described as a single-machine execution pipeline.

## Final fixed-initialization experiment arms

The five final experimental arms are:

1. `Sphere`
2. `Random`
3. `Original150k`
4. `Curv0`
5. `Curv1`

### Sphere

Final training experiment:

`exp37_hcpya+oasis1_onlyMRI_icosphere_clean_2gpu`

Fixed template:

`v2c_icosphere_fixed_template`

The spherical template was generated as a canonical subdivision-7 icosphere.

### Random

Final training experiment:

`exp36_hcpya+oasis1_onlyMRI_oasis_sub0447_unsmoothed_2gpu`

Fixed template:

`v2c_oasis_random_unsmoothed_seed2025_fixed_template`

The selected OASIS1 training subject was `sub-0447`, using selection seed `2025`.

### Original150k

Final training experiment:

`exp41_hcpya+oasis1_onlyMRI_sub298051_taubin150000_3gpu`

Fixed template:

`v2c_sub298051_taubin150000_fixed_template`

The source subject was `sub-298051`, selected as the lowest-population-mean white-matter ASSD medoid over the evaluated HCP population.

The fixed template was generated using ordinary Taubin smoothing for 150,000 iterations with the historically executed parameters:

* lambda: `0.5`
* nu: `0.5`

### Curv0

Final training experiment:

`exp42_hcpya+oasis1_onlyMRI_sub298051_caware150000_edgeonly062_raw_3gpu`

Fixed template:

`v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_fixed_template`

Its template lineage is:

```text
sub-298051 source surfaces
        |
        v
collision-aware Taubin smoothing to 150k
        |
        v
largest-connected-component cleanup
        |
        v
Geometry Central edge-length adjustment
target edge length = 0.618 mm
curvatureAdaptation = 0
```

### Curv1

Final training experiment:

`exp42_hcpya+oasis1_onlyMRI_sub298051_caware150000_edgeonly062_raw_3gpu_curv1`

Fixed template:

`v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_curv1_fixed_template`

Its template lineage matches Curv0 except that the Geometry Central remeshing runtime used:

`curvatureAdaptation = 1`

## Legacy exp35 sphere experiment

The earlier experiment historically named:

`exp35_v2c_icosphere`

also used:

`v2c_icosphere_fixed_template`

Despite the historical `v2c` naming, this run is an earlier icosphere fixed-initialization experiment.

It is not the final Sphere arm.

The final Sphere arm is `exp37_sphere_clean`.

For reproducibility documentation, the exp35 experiment is classified as:

`SUPERSEDED_BY_EXP37`

This historical naming must also not be confused with external Vertex2Cortex or V2C comparison baselines.

## Checkpoint provenance

The final inference checkpoints stored on Neuro-Ix were compared against their corresponding Narval training checkpoints using SHA-256.

All five final checkpoints matched byte-for-byte between the two systems.

The legacy exp35 sphere checkpoint also matched.

The verified checkpoint hashes are recorded in:

`manifests/reproducibility/checkpoints.tsv`

This establishes the following provenance chain:

```text
Narval training
      |
      | identical SHA-256 checkpoint
      v
Neuro-Ix inference
      |
      v
external evaluation
```

## External evaluation

The final external evaluation uses 14 datasets with 40 selected cases per dataset:

`14 x 40 = 560 cases per method`

The evaluation workflow includes:

* surface geometry metrics;
* scannerRAS conversion;
* exact FCL collision evaluation;
* surface-centric collision union summaries.

For completed final collision evaluations, exact `python-fcl` execution was verified on Neuro-Ix.

## Curv1 historical metadata discrepancy

One historical Curv1 `REMESH_METADATA.txt` record incorrectly reports:

`curvatureAdaptation=0`

This value is not authoritative for the actual Curv1 execution.

The corresponding Curv1 Geometry Central implementation and the recorded runtime log show that the executed setting was:

`curvatureAdaptation=1`

The runtime also recorded successful edge-length adjustment.

The historical metadata file is retained as historical evidence rather than silently modified. Reproducibility documentation should record the corrected runtime parameter explicitly.

## Data and artifact policy

Restricted or non-redistributable MRI datasets are not included in this repository.

The public reproducibility branch is intended to contain:

* source code;
* configuration files;
* template-generation scripts;
* remeshing scripts;
* machine-readable template metadata;
* SHA-256 manifests;
* inference code;
* evaluation code;
* reproducibility documentation.

Large generated fixed-template meshes and neural-network checkpoints may be distributed separately using an appropriate artifact mechanism such as Git LFS, GitHub Releases, or an archival repository.

Machine-specific forensic archives, private filesystem layouts, and restricted datasets are intentionally kept outside the public Git repository.

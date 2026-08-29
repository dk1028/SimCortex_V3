# Fixed-template provenance

This document records the provenance of the five final
MRI-only fixed-initialization templates.

Exact surface identities are pinned in:

`manifests/reproducibility/fixed_templates.tsv`

The large historical PLY files are not committed directly in this
repository.

## Hash evidence

For Sphere, Original150k, Curv0, and Curv1, the SHA256 values were
recovered from the Narval reproducibility inventory generated on
2026-08-28:

`reports/github_repro_inventory_20260828/narval/22_resource_sha256.txt`

That inventory hashed the historical resource files in the Narval
workspace.

For Random, the four canonical template hashes are preserved in the
historical:

`RANDOM_TEMPLATE_SHA256SUMS.txt`

The Mac forensic staging bundle contains template metadata and
provenance evidence but does not contain the large PLY files themselves.

## Sphere

Final arm:

`Sphere`

Final experiment:

`exp37_hcpya+oasis1_onlyMRI_icosphere_clean_2gpu`

Template:

`v2c_icosphere_fixed_template`

The final Sphere template was generated as a subdivision-7 icosphere.

Historical metadata records:

- subdivisions: 7
- center distance: 60.862778388599395 mm
- white radius: 24.88825027486973 mm
- pial radius: 27.38825027486973 mm
- white-pial radial gap: 2.5 mm
- expected left-right pial gap: 6.086277838859935 mm

Centers:

- LH: [-29.501429557800293, -28.34039306640625, 16.088603019714355]
- RH: [31.243325233459473, -24.554076194763184, 15.960684776306152]

Each of the four sphere surfaces has:

- 163842 vertices
- 327680 faces
- one connected component
- Euler number 2

The earlier exp35 sphere experiment is historical and superseded by
exp37 for the final Sphere arm.

## Random

Final arm:

`Random`

Final experiment:

`exp36_hcpya+oasis1_onlyMRI_oasis_sub0447_unsmoothed_2gpu`

Template:

`v2c_oasis_random_unsmoothed_seed2025_fixed_template`

Historical selection metadata records:

- dataset: OASIS1
- split: train
- selection seed: 2025
- selected subject: sub-0447
- session: ses-01
- coordinate space: MNI152
- source stage: sc-initsurf-0.2
- OASIS1 training subjects in the split: 291
- valid four-surface candidates: 291

The selected template is the unsmoothed subject-specific initialization
of OASIS1 sub-0447.

The four exact surface hashes are preserved in the historical
`RANDOM_TEMPLATE_SHA256SUMS.txt`.

## Original150k

Final arm:

`Original150k`

Final experiment:

`exp41_hcpya+oasis1_onlyMRI_sub298051_taubin150000_3gpu`

Template:

`v2c_sub298051_taubin150000_fixed_template`

Source subject:

`sub-298051`

The source subject was selected as the HCP white-matter medoid candidate
using left/right white-surface ASSD against the HCP population.

Historical smoothing metadata records:

- Taubin iterations: 150000
- lambda: +0.5
- nu: +0.5

The positive historical `nu=0.5` value is intentional provenance.
It must not be silently replaced with a conventional negative Taubin
parameter.

This arm is the ordinary, non-collision-aware 150k smoothing branch.

## Curv0

Final arm:

`Curv0`

Final experiment:

`exp42_hcpya+oasis1_onlyMRI_sub298051_caware150000_edgeonly062_raw_3gpu`

Template:

`v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_fixed_template`

The template lineage is:

1. sub-298051 source geometry
2. collision-aware Taubin smoothing to 150000
3. largest-connected-component cleanup
4. Geometry Central `adjustEdgeLengths()`
5. edge-only remeshing

Historical remeshing settings:

- target edge length: 0.618 mm
- curvature adaptation: 0
- maximum iterations: 10

The historical Curv0 generation log records the final four generated
surface meshes and their vertex/face counts.

## Curv1

Final arm:

`Curv1`

Final experiment:

`exp42_hcpya+oasis1_onlyMRI_sub298051_caware150000_edgeonly062_raw_3gpu_curv1`

Template:

`v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_curv1_fixed_template`

Curv1 uses the same collision-aware 150k and LCC-clean source lineage as
Curv0, followed by Geometry Central edge-length adjustment.

The actual Curv1 scientific setting is:

`curvatureAdaptation = 1.0`

### Historical metadata discrepancy

The historical Curv1 `REMESH_METADATA.txt` incorrectly records:

`curvatureAdaptation=0`

That historical metadata file is retained as provenance and must not be
silently edited.

The actual Curv1 C++ implementation sets
`options.curvatureAdaptation = 1.0`, and the historical runtime log for
job 1148038 records `curvatureAdaptation=1` while generating all four
Curv1 surfaces.

Therefore the implementation/runtime value 1.0 is the authoritative
scientific setting for Curv1.

## Reproduction contract

All five final deformation arms use:

- MRI only
- no ribbon probability-map input
- fixed initial cortical surfaces
- `model.c_in=1`

The template directory supplied to a reproduction run must contain the
canonical surface names:

- `lh_pial_smoothed.ply`
- `lh_white_smoothed.ply`
- `rh_pial_smoothed.ply`
- `rh_white_smoothed.ply`

Before training or inference, verify each file against
`manifests/reproducibility/fixed_templates.tsv`.

If the PLY assets are distributed separately, use an appropriate
large-file or archival distribution mechanism rather than ordinary Git.

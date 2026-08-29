# Reproducibility Index

This directory is the human-readable entry point for the fixed-initialization MRI-only SimCortex reproducibility branch.

## Scope

- Official upstream repository: `Neuro-iX/SimCortex`
- Pinned upstream baseline: `6f3cb21af9763807190407e9725331b0f2e78bed`
- Reproducibility branch: `repro/fixed-initialization`
- Historical final experiment family: MRI-only deformation with fixed cortical initialization

The upstream baseline is the current SimCortex v2.0 journal-version implementation. The older ShapeMI/MICCAI implementation is a separate legacy v1.0.0 code line and is not the baseline used here.

## What this branch reconstructs

The final experiment replaces the upstream deformation stage's subject-specific initialization dependency with one fixed four-surface cortical template and removes ribbon-probability conditioning from the deformation network.

Final deformation contract:

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

## Final experimental arms

The five final arms use the same MRI-only deformation architecture. Their principal experimental difference is the fixed initialization geometry.

| Arm | Fixed initialization |
|---|---|
| Sphere | canonical subdivision-7 icosphere |
| Random | OASIS1 `sub-0447`, deterministic seed-2025 selection |
| Original150k | `sub-298051` medoid geometry after ordinary 150k Taubin smoothing |
| Curv0 | collision-aware 150k lineage plus LCC cleanup and uniform edge-only remeshing |
| Curv1 | Curv0 lineage with curvature-adaptive remeshing (`curvatureAdaptation=1.0`) |

These are five initialization conditions, not five different neural-network architectures.

## Evidence map

| Question | Primary documentation | Machine-readable evidence | Status |
|---|---|---|---|
| What upstream revision is the reconstruction based on? | `BASELINE.md` | `source_archives.tsv` | PASS |
| What were the final five training configurations? | `BASELINE.md` | `training_configs.tsv` | PASS |
| Which exact final checkpoints were used? | `BASELINE.md` | `checkpoints.tsv` | PASS |
| Where did each fixed template come from? | `FIXED_TEMPLATES.md` | `fixed_templates.tsv` | PASS |
| Do the five final checkpoints strictly load into the clean MRI-only model? | `CHECKPOINT_COMPATIBILITY.md` | `checkpoint_compatibility.tsv` | PASS |
| Does the clean reconstruction run end-to-end on a real historical MRI? | `INFERENCE_REPRODUCTION.md` | `real_mri_smoke.tsv` | PASS |
| How does the experiment differ from official SimCortex v2.0? | `UPSTREAM_DELTA.md` | source/config history | DOCUMENTED |

## Reproducibility status

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

## Commit chain

The branch was reconstructed as a sequence of narrow, auditable commits:

```text
268bc0f  chore(repro): pin baseline and experiment provenance
8856393  feat(deform): add MRI-only deformation architecture
30a17eb  feat(data): support MRI-only fixed initialization
b1f303a  feat(train): wire MRI-only fixed initialization
46e64b3  feat(infer): wire MRI-only fixed initialization
56dfc56  feat(config): align deformation defaults with fixed initialization
52fdb0f  feat(repro): add five-arm training overlays
7a57ea7  docs(repro): pin fixed-template surface provenance
911f310  docs(repro): record checkpoint compatibility audit
3d4e5b1  docs(repro): record real-MRI inference smoke test
```

The scientific execution claims are pinned to the commits and hashes recorded in the corresponding documents. Later documentation commits do not retroactively change those tested artifacts.

## Data and artifact boundary

Restricted MRI datasets, historical private filesystem payloads, large checkpoints, and large template meshes are not committed to ordinary Git.

The repository instead records:

- source and configuration needed to reconstruct the experiment;
- exact SHA256 identities for historical checkpoints and fixed templates;
- machine-readable manifests;
- provenance and execution evidence;
- limitations on what was and was not replayed.

## Important limitation

The clean Narval real-MRI smoke test is a functional reproduction, not an exact replay of the historical Neuro-Ix sample40 external evaluation.

The Neuro-Ix inference launchers and path provenance were captured, but the exact historical Neuro-Ix MRI/preprocessing payload was not preserved in the final local snapshot before access ended. Therefore the repository intentionally does not claim an exact sample40 replay.

# Final checkpoint compatibility

This document records a read-only compatibility audit of the five final
MRI-only fixed-initialization checkpoints against the clean reproducibility
implementation.

The machine-readable audit result is pinned in:

`manifests/reproducibility/checkpoint_compatibility.tsv`

The exact final checkpoint identities are pinned separately in:

`manifests/reproducibility/checkpoints.tsv`

## Audited clean commit

The compatibility audit was executed against:

`7a57ea72d4235baeeda503845325358d09ae6951`

Commit subject:

`docs(repro): pin fixed-template surface provenance`

The audit used a separate disposable clone on Narval. The clone was checked
out detached at the exact commit above.

After the audit:

- `git status --short` produced no output
- `git rev-parse HEAD` remained
  `7a57ea72d4235baeeda503845325358d09ae6951`

Therefore the audited clean source tree was not modified by the test.

## Environment

The checkpoint audit used the historical/reproducibility Python environment:

`/scratch/kimdowoo/simcortex_env_py310_ready/bin/python`

Observed PyTorch version:

`2.1.0`

The source package was imported directly from the clean clone through its
`src` directory.

The authoritative model import was:

`from simcortex.deform.models.surfdeform import SurfDeform`

## Final arms

The following final checkpoints were audited:

- Sphere
- Random
- Original150k
- Curv0
- Curv1

The historical exp35 sphere checkpoint was not treated as a final arm.
It remains superseded by the final exp37 Sphere checkpoint.

## Checkpoint serialization

All five final `deform_best_rmse.pth` checkpoints were observed as raw:

`OrderedDict`

state dictionaries.

Each checkpoint contained:

`108`

model-state keys.

The final checkpoints did not require a `state_dict`, `model`, or
`model_state_dict` wrapper to recover the model parameters.

All five checkpoints used the same parameter namespace and tensor-shape
structure.

## MRI-only architecture evidence

For every final checkpoint, the first convolution tensor was:

`munet.m1.0.conv.weight`

with shape:

`(8, 1, 3, 3, 3)`

The input-channel dimension is therefore one, consistent with the final
MRI-only deformation contract:

- MRI input only
- `model.c_in=1`
- no ribbon probability-map input
- fixed initial cortical surfaces

## Structural compatibility

For each of the five final checkpoints, the audit instantiated the clean
`SurfDeform` model using the clean inference configuration:

- `c_in=1`
- `c_hid=[8,16,32,64,128,128]`
- `inshape=[184,224,184]`
- `sigma=1`
- `gn_groups=8`
- `dropout=0.1`

`n_steps=8` is an inference/deformation integration argument rather than a
`SurfDeform` constructor parameter.

For every final arm:

- clean model keys: 108
- checkpoint keys: 108
- missing keys: 0
- unexpected keys: 0
- tensor-shape mismatches: 0
- namespace matched the Sphere reference

## Strict loading

For every final arm, direct:

`model.load_state_dict(..., strict=True)`

completed successfully with:

- zero missing keys
- zero unexpected keys

The clean production inference helper:

`simcortex.deform.inference.load_checkpoint`

was then tested separately for each exact final checkpoint with:

`strict=True`

All five production-loader calls succeeded.

## Checkpoint byte identity

Before compatibility testing, each checkpoint was verified against the
previously pinned final checkpoint identity.

For all five final arms:

- file size matched
- SHA256 matched

The exact checkpoint sizes and SHA256 values are recorded in:

`manifests/reproducibility/checkpoints.tsv`

## Audit result

The five final checkpoints are structurally and strictly load-compatible
with the clean reproducibility implementation at commit:

`7a57ea72d4235baeeda503845325358d09ae6951`

The final audit result was:

`FIVE_FINAL_CHECKPOINTS_COMPATIBLE = True`

No checkpoint-loader compatibility patch was required.

In particular, support for a `model_state_dict` wrapper is not required by
the five verified final checkpoints because each is a raw `OrderedDict`
state dictionary.

## Raw audit evidence

The original Narval-generated TSV was copied byte-for-byte into:

`manifests/reproducibility/checkpoint_compatibility.tsv`

Its SHA256 is:

`978d86e8285fb8b3da6ef71a26965c0e60affef2679e0f84ee630dcc2e610269`

This hash was verified again after transferring the audit report from
Narval to the Mac forensic evidence archive.

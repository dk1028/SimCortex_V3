# Fixed Template Resources

The deformation pipeline requires one fixed four-surface initialization template for each experimental arm.

Set:

```bash
export SIMCORTEX_RESOURCES_ROOT=/path/to/resources
```

The expected structure is:

```text
$SIMCORTEX_RESOURCES_ROOT/
  v2c_icosphere_fixed_template/
  v2c_oasis_random_unsmoothed_seed2025_fixed_template/
  v2c_sub298051_taubin150000_fixed_template/
  v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_fixed_template/
  v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_curv1_fixed_template/
```

Each directory must contain:

```text
lh_pial_smoothed.ply
lh_white_smoothed.ply
rh_pial_smoothed.ply
rh_white_smoothed.ply
```

Arm mapping:

| Arm | Directory |
|---|---|
| Sphere | `v2c_icosphere_fixed_template` |
| Random | `v2c_oasis_random_unsmoothed_seed2025_fixed_template` |
| Original150k | `v2c_sub298051_taubin150000_fixed_template` |
| Curv0 | `v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_fixed_template` |
| Curv1 | `v2c_sub298051_collisionaware_taubin150000_lccclean_gc_edgeonly062_curv1_fixed_template` |

Exact SHA256 identities and mesh topology are recorded in:

```text
manifests/reproducibility/fixed_templates.tsv
docs/reproducibility/FIXED_TEMPLATES.md
```

The MRI datasets are not template resources and are not distributed here.

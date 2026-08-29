import os
import gc
import math
import logging
import tempfile
from datetime import timedelta
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple
import random
import numpy as np

import hydra
from hydra.utils import to_absolute_path
import torch
import torch.nn.functional as F
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_normal_consistency
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import knn_points

from simcortex.deform.data.dataloader import CSRDeformDataset, collate_csr_deform
from simcortex.deform.utils.coords import voxel_to_world
from simcortex.deform.models.surfdeform import SurfDeform

import trimesh

# Single source of truth for collisions: the same backend the evaluator uses.
# Place collision_backend.py on PYTHONPATH (e.g. next to this file or inside
# the simcortex package). It depends only on numpy + trimesh + fcl.
from simcortex.utils.collision_backend import HAS_FCL, collision_pair_from_meshes


log = logging.getLogger(__name__)



def vertex_normals_from_mesh(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    verts: (V, 3)
    faces: (F, 3)
    returns vertex normals: (V, 3)
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = F.normalize(fn, dim=-1, eps=1e-8)

    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    vn = F.normalize(vn, dim=-1, eps=1e-8)
    return vn

def signed_nested_surface_penalty(
    white_v: torch.Tensor,
    white_f: torch.Tensor,
    pial_v: torch.Tensor,
    pial_f: torch.Tensor,
    margin_mm: float = 0.5,
    n_points: int = 40000,
):
    """
    Enforces:
      - white should stay inside pial
      - pial should stay outside white

    Sign convention (assuming outward normals):
      - point inside a closed surface  -> signed distance < 0
      - point outside a closed surface -> signed distance > 0

    NOTE: sign is determined by nearest-vertex normal projection rather than an
    exact signed distance or winding number. It is therefore a local nesting proxy
    and can be inaccurate near sulci or other highly curved non-convex regions.
    Report this approximation and its configured weight explicitly in experiments.
    """
    device = white_v.device

    mesh_w = Meshes(verts=[white_v], faces=[white_f])
    mesh_p = Meshes(verts=[pial_v], faces=[pial_f])

    # surface samples instead of vertex-only samples
    w_pts = sample_points_from_meshes(mesh_w, num_samples=n_points).squeeze(0)
    p_pts = sample_points_from_meshes(mesh_p, num_samples=n_points).squeeze(0)

    pial_normals = vertex_normals_from_mesh(pial_v, pial_f)
    white_normals = vertex_normals_from_mesh(white_v, white_f)

    # ---------------------------------------------------
    # 1) white samples relative to pial
    # white should be INSIDE pial => signed_w should be <= -margin
    # ---------------------------------------------------
    knn_wp = knn_points(w_pts[None], pial_v[None], K=1, return_nn=False)
    idx_pial = knn_wp.idx[0, :, 0]
    nearest_pial = pial_v[idx_pial]
    nearest_pial_n = pial_normals[idx_pial]

    signed_w = ((w_pts - nearest_pial) * nearest_pial_n).sum(dim=-1)
    loss_w = F.relu(signed_w + margin_mm).mean()

    # ---------------------------------------------------
    # 2) pial samples relative to white
    # pial should be OUTSIDE white => signed_p should be >= +margin
    # ---------------------------------------------------
    knn_pw = knn_points(p_pts[None], white_v[None], K=1, return_nn=False)
    idx_white = knn_pw.idx[0, :, 0]
    nearest_white = white_v[idx_white]
    nearest_white_n = white_normals[idx_white]

    signed_p = ((p_pts - nearest_white) * nearest_white_n).sum(dim=-1)
    loss_p = F.relu(margin_mm - signed_p).mean()

    loss = 0.5 * (loss_w + loss_p)

    with torch.no_grad():
        bad_white_pct = (signed_w > -margin_mm).float().mean().item() * 100.0
        bad_pial_pct = (signed_p < margin_mm).float().mean().item() * 100.0
        mean_signed_w = signed_w.mean().item()
        mean_signed_p = signed_p.mean().item()

    return loss, bad_white_pct, bad_pial_pct, mean_signed_w, mean_signed_p

def count_collisions_inmemory(
    vA_mm: torch.Tensor, fA: torch.Tensor,
    vB_mm: torch.Tensor, fB: torch.Tensor
):
    """
    vA_mm, vB_mm: (V,3) torch float in mm-space (GPU/CPU)
    fA, fB: (F,3) torch long
    Returns: (is_col: bool or None, n_contacts: int or None)

    Uses the shared collision_backend (direct python-fcl), so the contact count
    is the raw FCL triangle-contact count -- not the degenerate ~1
    that trimesh's CollisionManager.in_collision_internal returns. The boolean
    is_col is unchanged vs the old path, so collision-rate-based model selection
    is identical to before; only the magnitude is now meaningful.
    """
    if not HAS_FCL:
        return None, None

    vA = vA_mm.detach().float().cpu().numpy()
    vB = vB_mm.detach().float().cpu().numpy()
    fA_np = fA.detach().long().cpu().numpy()
    fB_np = fB.detach().long().cpu().numpy()

    if vA.shape[0] == 0 or vB.shape[0] == 0 or fA_np.shape[0] == 0 or fB_np.shape[0] == 0:
        return False, 0

    mA = trimesh.Trimesh(vertices=vA, faces=fA_np, process=False)
    mB = trimesh.Trimesh(vertices=vB, faces=fB_np, process=False)

    res = collision_pair_from_meshes(mA, mB)
    if res["fcl_status"] != "OK":
        log.error(
            "FCL collision query failed: status=%s error=%s",
            res.get("fcl_status"),
            res.get("fcl_error", ""),
        )
        return None, None
    n_contacts = int(res["num_contacts"])
    return (n_contacts > 0), n_contacts


# -----------------------
# DDP helpers
# -----------------------
def setup_ddp() -> Tuple[int, int, int, bool]:
    """Return (rank, world_size, local_rank, is_distributed)."""
    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if has_rank != has_world_size:
        raise RuntimeError("RANK and WORLD_SIZE must either both be set or both be absent")

    if has_rank:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training uses NCCL and therefore requires CUDA")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world_size <= 0 or not (0 <= rank < world_size):
            raise ValueError(f"Invalid DDP rank/world_size: rank={rank}, world_size={world_size}")
        if not (0 <= local_rank < torch.cuda.device_count()):
            raise ValueError(
                f"Invalid LOCAL_RANK={local_rank}; visible CUDA devices={torch.cuda.device_count()}"
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=6),
        )
        return rank, world_size, local_rank, True

    return 0, 1, 0, False


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_map(cfg_node, keys: Tuple[str, ...]) -> Dict[str, str] | None:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            return {str(kk): str(vv) for kk, vv in v.items()}
    return None


def _as_abs_path(path_like: Any, *, field: str) -> str:
    value = str(path_like).strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return to_absolute_path(value)


def _optional_abs_path(path_like: Any) -> str:
    value = str(path_like or "").strip()
    return to_absolute_path(value) if value else ""


def _require_finite(name: str, value: Any, *, minimum=None, maximum=None) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {number}")
    return number


def validate_deform_training_config(cfg: DictConfig) -> None:
    required_surfaces = {"lh_pial", "lh_white", "rh_pial", "rh_white"}
    surface_names = [str(name).strip() for name in cfg.dataset.surface_name]
    if len(surface_names) != 4 or set(surface_names) != required_surfaces:
        raise ValueError(
            "dataset.surface_name must contain each of lh_pial, lh_white, "
            f"rh_pial, and rh_white exactly once; got {surface_names}"
        )
    if len(surface_names) != len(set(surface_names)):
        raise ValueError(f"dataset.surface_name contains duplicates: {surface_names}")

    inshape = tuple(int(value) for value in cfg.model.inshape)
    if len(inshape) != 3 or any(value <= 0 or value % 8 != 0 for value in inshape):
        raise ValueError(
            "model.inshape must contain three positive dimensions divisible by 8, "
            f"got {inshape}"
        )
    channels = tuple(int(value) for value in cfg.model.c_hid)
    if len(channels) != 6 or any(value <= 0 for value in channels):
        raise ValueError(f"model.c_hid must contain six positive integers, got {channels}")
    use_probability_map = bool(
        OmegaConf.select(
            cfg,
            "dataset.use_probability_map",
            default=True,
        )
    )
    if use_probability_map:
        raise ValueError(
            "MRI-only training requires "
            "dataset.use_probability_map=false."
        )

    if int(cfg.model.c_in) != 1:
        raise ValueError(
            "MRI-only training requires model.c_in=1, "
            f"but got {cfg.model.c_in}."
        )

    use_fixed_initial_surface = bool(
        OmegaConf.select(
            cfg,
            "dataset.use_fixed_initial_surface",
            default=False,
        )
    )
    fixed_template_root = OmegaConf.select(
        cfg,
        "dataset.fixed_template_root",
        default=None,
    )
    if (
        use_fixed_initial_surface
        and fixed_template_root in (None, "")
    ):
        raise ValueError(
            "dataset.use_fixed_initial_surface=true requires "
            "dataset.fixed_template_root."
        )
    if int(cfg.model.n_steps) < 0:
        raise ValueError(f"model.n_steps must be >= 0, got {cfg.model.n_steps}")
    _require_finite("model.sigma", cfg.model.sigma, minimum=1e-12)
    _require_finite("model.dropout", cfg.model.dropout, minimum=0.0, maximum=0.999999)

    positive_ints = {
        "trainer.img_batch_size": cfg.trainer.img_batch_size,
        "trainer.grad_accum_steps": cfg.trainer.grad_accum_steps,
        "trainer.num_epochs": cfg.trainer.num_epochs,
        "trainer.validation_interval": cfg.trainer.validation_interval,
        "trainer.collision_interval": cfg.trainer.collision_interval,
        "trainer.points_per_image": cfg.trainer.points_per_image,
        "trainer.val_points_per_image": cfg.trainer.val_points_per_image,
        "trainer.mesh_chunk": cfg.trainer.mesh_chunk,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if int(cfg.trainer.num_workers) < 0:
        raise ValueError(f"trainer.num_workers must be >= 0, got {cfg.trainer.num_workers}")
    if int(getattr(cfg.trainer, "prefetch_factor", 2)) <= 0:
        raise ValueError("trainer.prefetch_factor must be > 0")
    if int(cfg.trainer.scheduler_patience) < 0:
        raise ValueError("trainer.scheduler_patience must be >= 0")
    if int(cfg.trainer.scheduler_cooldown) < 0:
        raise ValueError("trainer.scheduler_cooldown must be >= 0")
    if int(getattr(cfg.model, "gn_groups", 8)) <= 0:
        raise ValueError("model.gn_groups must be > 0")
    _require_finite("trainer.learning_rate", cfg.trainer.learning_rate, minimum=1e-12)
    _require_finite("trainer.weight_decay", cfg.trainer.weight_decay, minimum=0.0)
    _require_finite("trainer.grad_clip_norm", cfg.trainer.grad_clip_norm, minimum=0.0)
    _require_finite("trainer.scheduler_factor", cfg.trainer.scheduler_factor, minimum=1e-12, maximum=1.0)
    _require_finite("trainer.scheduler_min_lr", cfg.trainer.scheduler_min_lr, minimum=0.0)
    _require_finite("trainer.scheduler_threshold_mm", cfg.trainer.scheduler_threshold_mm, minimum=0.0)
    _require_finite("trainer.early_stop_min_delta_mm", cfg.trainer.early_stop_min_delta_mm, minimum=0.0)
    if int(cfg.trainer.early_stop_patience) < 0:
        raise ValueError("trainer.early_stop_patience must be >= 0")

    if not bool(getattr(cfg.dataset, "strict_missing", True)):
        raise ValueError(
            "dataset.strict_missing must be true for deformation training so the split "
            "cohort cannot change silently"
        )

    clip_min = _require_finite("dataset.prob_clip_min", cfg.dataset.prob_clip_min, minimum=0.0, maximum=1.0)
    clip_max = _require_finite("dataset.prob_clip_max", cfg.dataset.prob_clip_max, minimum=0.0, maximum=1.0)
    if clip_min > clip_max:
        raise ValueError("dataset.prob_clip_min must be <= dataset.prob_clip_max")
    _require_finite("dataset.prob_gamma", cfg.dataset.prob_gamma, minimum=1e-12)
    for name in ("aug_prob", "aug_intensity_prob"):
        _require_finite(f"dataset.{name}", getattr(cfg.dataset, name), minimum=0.0, maximum=1.0)
    for name in (
        "aug_rot_range_deg", "aug_scale_range", "aug_trans_range_mm",
        "aug_bias_strength", "aug_gain_range", "aug_bright_range", "aug_noise_std",
    ):
        _require_finite(f"dataset.{name}", getattr(cfg.dataset, name), minimum=0.0)

    for name in (
        "chamfer_weight", "chamfer_scale", "edge_loss_weight", "normal_weight",
        "hd_weight", "hd_lambda_mm", "signed_nested_weight", "signed_margin_mm",
        "pial_lr_hd_weight", "pial_lr_hd_lambda_mm",
    ):
        _require_finite(f"objective.{name}", getattr(cfg.objective, name), minimum=0.0)
    for name in ("hd_p", "pial_lr_hd_p"):
        _require_finite(f"objective.{name}", getattr(cfg.objective, name), minimum=0.0, maximum=1.0)
    for name in ("hd_points", "signed_points", "pial_lr_hd_points"):
        if int(getattr(cfg.objective, name)) <= 0:
            raise ValueError(f"objective.{name} must be > 0")
    if int(cfg.objective.reg_warmup_epochs) < 0:
        raise ValueError("objective.reg_warmup_epochs must be >= 0")

    for name in ("alpha_wp", "alpha_lr", "min_delta_score"):
        _require_finite(f"checkpoint.{name}", getattr(cfg.checkpoint, name), minimum=0.0)
    _require_finite("checkpoint.rmse_guardrail_rel", cfg.checkpoint.rmse_guardrail_rel, minimum=1.0)

    val_interval = int(cfg.trainer.validation_interval)
    collision_interval = int(cfg.trainer.collision_interval)
    if bool(cfg.checkpoint.require_collision_for_best) and collision_interval != val_interval:
        raise ValueError(
            "checkpoint.require_collision_for_best=True requires "
            "trainer.collision_interval == trainer.validation_interval; "
            f"got {collision_interval} and {val_interval}"
        )


def validate_split_dataframe(df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    df = normalize_subject_column(df)
    if mode == "multi":
        _validate_multi_split_df(df)
        df["dataset"] = df["dataset"].astype(str).str.strip()
        invalid_dataset = df["dataset"].str.lower().isin({"", "nan", "none"})
        if invalid_dataset.any():
            raise ValueError("split_file contains empty/invalid dataset labels")
        duplicate_subset = ["dataset", "subject"]
    elif mode == "single":
        _validate_single_split_df(df)
        duplicate_subset = ["subject"]
    else:
        raise ValueError(f"Unknown deform training mode: {mode}")

    df["split"] = df["split"].astype(str).str.strip()
    invalid_split = df["split"].str.lower().isin({"", "nan", "none"})
    if invalid_split.any():
        raise ValueError("split_file contains empty/invalid split labels")

    duplicates = df.duplicated(subset=duplicate_subset, keep=False)
    if duplicates.any():
        columns = duplicate_subset + ["split"]
        raise ValueError(
            "split_file contains duplicate subject rows:\n"
            + df.loc[duplicates, columns].sort_values(duplicate_subset).to_string(index=False)
        )
    return df


def _load_trusted_checkpoint(path: str, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch versions before weights_only was added.
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(obj: Any, path: str) -> None:
    parent = os.path.dirname(path)
    if not parent:
        raise ValueError(f"Checkpoint path must include a parent directory: {path}")
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


def _validate_output_root(out_root: str, *, resume_from: str, allow_existing: bool) -> None:
    if not out_root:
        raise ValueError("outputs.root must not be empty")
    if resume_from or allow_existing or not os.path.isdir(out_root):
        return
    existing = sorted(os.listdir(out_root))
    if existing:
        preview = existing[:20]
        raise FileExistsError(
            f"Refusing to start a fresh run in non-empty training directory {out_root}. "
            f"Existing entries: {preview}. Use a new outputs.root, resume from a full "
            "checkpoint, or explicitly set outputs.allow_existing=true."
        )


def _preflight_dataset(dataset, *, label: str, expected_surfaces: List[str]) -> None:
    for index in tqdm(range(len(dataset)), desc=f"Preflight {label}", unit="subj"):
        sample = dataset[index]
        subject = sample.get("subject", f"index-{index}")
        vol = sample.get("vol")
        if not torch.is_tensor(vol) or vol.ndim != 4 or int(vol.shape[0]) != 1:
            raise ValueError(
                f"{label} {subject}: expected MRI-only vol shape "
                f"(1,D,H,W), got {getattr(vol, 'shape', None)}"
            )
        if not torch.isfinite(vol).all():
            raise ValueError(f"{label} {subject}: volume contains non-finite values")
        for field in ("init_verts_vox", "init_faces", "gt_verts_vox", "gt_faces"):
            values = sample.get(field, {})
            if set(values) != set(expected_surfaces):
                raise ValueError(
                    f"{label} {subject}: {field} keys={sorted(values)}, expected={sorted(expected_surfaces)}"
                )
        if (index + 1) % 25 == 0:
            gc.collect()


def _rank0_preflight_or_raise(train_ds, val_ds, *, rank: int, is_distributed: bool, surface_names: List[str]) -> None:
    payload = [""]
    if rank == 0:
        try:
            _preflight_dataset(train_ds, label="train", expected_surfaces=surface_names)
            _preflight_dataset(val_ds, label="validation", expected_surfaces=surface_names)
        except Exception as exc:
            payload[0] = f"Deformation dataset preflight failed: {type(exc).__name__}: {exc}"
    if is_distributed:
        dist.broadcast_object_list(payload, src=0)
    if payload[0]:
        raise RuntimeError(payload[0])


def _raise_if_any_rank_error(messages: List[str], *, device, is_distributed: bool, context: str) -> None:
    flag = torch.tensor(1 if messages else 0, device=device, dtype=torch.int64)
    if is_distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        detail = "; ".join(messages[:8]) if messages else "detected on another DDP rank"
        raise RuntimeError(f"{context}: {detail}")


def validation_coverage_status(val_count, val_surf, *, expected_subjects: int, surface_names: List[str]):
    expected_total = int(expected_subjects) * len(surface_names)
    problems = []
    if int(val_count) != expected_total:
        problems.append(f"total surfaces={int(val_count)}/{expected_total}")
    for surface in surface_names:
        observed = int(val_surf[surface]["count"])
        if observed != int(expected_subjects):
            problems.append(f"{surface}={observed}/{int(expected_subjects)}")
    return (not problems), ", ".join(problems)


def collision_coverage_status(lh_total, rh_total, lr_total, lh_unknown, rh_unknown, lr_unknown, *, expected_subjects: int):
    expected = int(expected_subjects)
    problems = []
    for label, total, unknown in (
        ("LH white-pial", lh_total, lh_unknown),
        ("RH white-pial", rh_total, rh_unknown),
        ("LR pial-pial", lr_total, lr_unknown),
    ):
        if int(total) != expected or int(unknown) != 0:
            problems.append(f"{label}: valid={int(total)}/{expected}, unknown={int(unknown)}")
    return (not problems), ", ".join(problems)


def normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        raise ValueError(f"split_file must contain column 'subject'. Got: {list(df.columns)}")
    df["subject"] = df["subject"].astype(str).str.strip()
    invalid = df["subject"].str.lower().isin({"", "nan", "none", "sub-"})
    if invalid.any():
        raise ValueError("split_file contains empty/invalid subject labels")
    df["subject"] = df["subject"].apply(
        lambda value: value if value.startswith("sub-") else f"sub-{value}"
    )
    return df


def _validate_single_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Single-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    if "dataset" in df.columns:
        vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                "Single-dataset deform train received a split_file with multiple dataset values. "
                "Please provide a split CSV for one dataset only."
            )


def _validate_multi_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Multi-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")


def _requires_subject_initsurf(cfg) -> bool:
    """
    Return whether the selected input mode needs subject-specific
    initialization-surface resources.

    Subject-specific resources are required when either:
      1. the ribbon probability map is used, or
      2. initial surfaces are subject-specific.

    MRI-only fixed-template training needs neither.
    """
    use_probability_map = bool(
        OmegaConf.select(
            cfg,
            "dataset.use_probability_map",
            default=True,
        )
    )
    use_fixed_initial_surface = bool(
        OmegaConf.select(
            cfg,
            "dataset.use_fixed_initial_surface",
            default=False,
        )
    )

    return (
        use_probability_map
        or not use_fixed_initial_surface
    )


def _detect_deform_train_mode(cfg):
    single_preproc_root = OmegaConf.select(
        cfg,
        "dataset.path",
        default=None,
    )
    single_initsurf_root = OmegaConf.select(
        cfg,
        "dataset.initsurf_root",
        default=None,
    )

    single_preproc_root = (
        None
        if single_preproc_root in (None, "")
        else str(single_preproc_root)
    )
    single_initsurf_root = (
        None
        if single_initsurf_root in (None, "")
        else str(single_initsurf_root)
    )

    roots_map = _get_map(
        cfg.dataset,
        ("roots",),
    )
    initsurf_roots_map = _get_map(
        cfg.dataset,
        ("initsurf_roots",),
    )

    needs_subject_initsurf = _requires_subject_initsurf(cfg)

    if single_preproc_root is not None:
        log.info("Deform training mode: SINGLE-DATASET")
        log.info("dataset.path = %s", single_preproc_root)
        log.info(
            "subject-specific initsurf required = %s",
            needs_subject_initsurf,
        )

        if needs_subject_initsurf and single_initsurf_root is None:
            raise ValueError(
                "Single-dataset training requires "
                "dataset.initsurf_root for the selected input mode."
            )

        if roots_map is not None:
            log.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )

        if initsurf_roots_map is not None:
            log.warning(
                "Both dataset.initsurf_root and "
                "dataset.initsurf_roots are present. "
                "Using SINGLE-DATASET mode and ignoring "
                "dataset.initsurf_roots."
            )

        return (
            "single",
            single_preproc_root,
            single_initsurf_root,
            None,
            None,
        )

    if roots_map is not None:
        log.info("Deform training mode: MULTI-DATASET")
        log.info(
            "dataset.roots keys = %s",
            list(roots_map.keys()),
        )
        log.info(
            "subject-specific initsurf required = %s",
            needs_subject_initsurf,
        )

        if needs_subject_initsurf and initsurf_roots_map is None:
            raise ValueError(
                "Multi-dataset training requires "
                "dataset.initsurf_roots for the selected input mode."
            )

        if initsurf_roots_map is not None:
            missing = sorted(
                set(roots_map.keys())
                - set(initsurf_roots_map.keys())
            )

            if needs_subject_initsurf and missing:
                raise KeyError(
                    "dataset.initsurf_roots is missing keys "
                    f"required by dataset.roots: {missing}"
                )

            extra = sorted(
                set(initsurf_roots_map.keys())
                - set(roots_map.keys())
            )

            if extra:
                log.warning(
                    "dataset.initsurf_roots has extra keys "
                    "not present in dataset.roots: %s",
                    extra,
                )

        return (
            "multi",
            None,
            None,
            roots_map,
            initsurf_roots_map,
        )

    raise ValueError(
        "Could not determine deform training mode. Provide either:\n"
        "  - dataset.path for single-dataset mode, or\n"
        "  - dataset.roots for multi-dataset mode.\n"
        "An initsurf root is required only when the selected "
        "input mode uses subject-specific initialization resources."
    )


# -----------------------
# Geometry helpers
# -----------------------
def mesh_is_valid(verts: torch.Tensor, faces: torch.Tensor) -> bool:
    if verts is None or faces is None:
        return False
    if verts.ndim != 2 or faces.ndim != 2:
        return False
    if verts.shape[1] != 3 or faces.shape[1] != 3:
        return False
    if verts.numel() == 0 or faces.numel() == 0:
        return False
    if torch.isnan(verts).any() or torch.isinf(verts).any():
        return False
    f = faces.long()
    if f.min().item() < 0:
        return False
    if f.max().item() >= verts.shape[0]:
        return False
    return True


def edge_length_preservation_loss(
    pred_v: torch.Tensor,
    init_v: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizes deviation of predicted edge lengths from the corresponding
    initial-mesh edge lengths.

    pred_v: (V, 3) predicted vertices in mm
    init_v: (V, 3) initial vertices in mm (same topology as pred_v)
    faces:  (F, 3) long face indices shared by pred_v and init_v

    Note: all three triangle edges are used. Duplicate edges shared by adjacent
    faces are intentionally harmless because pred/init are weighted identically.
    """
    f = faces.long()
    edges = torch.cat(
        [
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [2, 0]],
        ],
        dim=0,
    )

    len_pred = (pred_v[edges[:, 0]] - pred_v[edges[:, 1]]).norm(dim=-1)
    with torch.no_grad():
        len_init = (init_v[edges[:, 0]] - init_v[edges[:, 1]]).norm(dim=-1)

    return ((len_pred - len_init) ** 2).mean()


# -----------------------
# Low-quantile separation penalty
# -----------------------
_PointFaceDistanceOP = _PointFaceDistance.apply


def point_to_mesh_dist_p3d(points: torch.Tensor, mesh: Meshes) -> torch.Tensor:
    """
    points: (N,3) float on device
    mesh: Meshes (batch size 1)
    returns: (N,) distances in same units as verts (here mm)
    """
    pts = points
    first_idx = torch.zeros((1,), device=pts.device, dtype=torch.int64)  # batch size 1
    max_pts = int(pts.shape[0])

    tris = mesh.verts_packed()[mesh.faces_packed()]  # (F,3,3)
    tri_first = mesh.mesh_to_faces_packed_first_idx()  # (1,)

    d2 = _PointFaceDistanceOP(pts, first_idx, tris, tri_first, max_pts)  # squared
    return d2.clamp_min(0.0).sqrt()


def partial_hd_penalty(mesh_a: Meshes, mesh_b: Meshes, p: float, lam: float, n_pts: int):
    """
    Low-quantile symmetric separation penalty.

    This is not classical Hausdorff distance. It samples points on both meshes,
    computes symmetric point-to-surface distances, takes a LOW quantile, and
    penalizes it if the separation is below lam.

    Returns:
      sep_q_mm: scalar tensor (mm)
      penalty: scalar tensor = relu(lam - sep_q_mm)
    """
    pa = sample_points_from_meshes(mesh_a, num_samples=n_pts).squeeze(0)
    pb = sample_points_from_meshes(mesh_b, num_samples=n_pts).squeeze(0)

    da = point_to_mesh_dist_p3d(pa, mesh_b)
    db = point_to_mesh_dist_p3d(pb, mesh_a)

    d_all = torch.cat([da, db], dim=0)  # (2n,)
    sep_q_mm = torch.quantile(d_all, q=float(p))

    lam_t = sep_q_mm.new_tensor(float(lam))
    penalty = F.relu(lam_t - sep_q_mm)
    return sep_q_mm, penalty


# -----------------------
# Random affine augmentation in NDC (volume + verts)
# -----------------------
def voxel_sizes_xyz_from_affine(A: torch.Tensor) -> torch.Tensor:
    A3 = A[:3, :3]
    vsize_ijk = torch.linalg.norm(A3, dim=0).clamp(min=1e-6)
    return vsize_ijk[[2, 1, 0]]  # xyz


def ijk_to_xyz(v_ijk: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_ijk[..., 2], v_ijk[..., 1], v_ijk[..., 0]], dim=-1)


def xyz_to_ijk(v_xyz: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_xyz[..., 2], v_xyz[..., 1], v_xyz[..., 0]], dim=-1)


def voxel_to_ndc_xyz(v_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=v_xyz.device, dtype=v_xyz.dtype).clamp(min=1.0)
    return 2.0 * (v_xyz / den) - 1.0


def ndc_to_voxel_xyz(u_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=u_xyz.device, dtype=u_xyz.dtype).clamp(min=1.0)
    return 0.5 * (u_xyz + 1.0) * den


def random_affine_ndc_xyz(B: int, rot_deg: float, scale_range: float, trans_ndc_xyz: torch.Tensor, device, dtype):
    ang = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * (rot_deg * math.pi / 180.0)
    cx, sx = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
    cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
    cz, sz = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])

    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx, cx
    ], dim=-1).view(-1, 3, 3)

    Ry = torch.stack([
        cy, torch.zeros_like(cy), sy,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy), cy
    ], dim=-1).view(-1, 3, 3)

    Rz = torch.stack([
        cz, -sz, torch.zeros_like(cz),
        sz, cz, torch.zeros_like(cz),
        torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=-1).view(-1, 3, 3)

    R = Rz @ Ry @ Rx

    ds = (torch.rand(B, 1, device=device, dtype=dtype) * 2 - 1) * scale_range
    s = 1.0 + ds
    A = R * s.view(B, 1, 1)

    t = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * trans_ndc_xyz
    b = t
    return A, b


def apply_aug(vol, padded_init_ijk, lengths, gt_verts_dict_list, affines, cfg, surface_names):
    prob = float(getattr(cfg.dataset, "aug_prob", 0.0))
    if prob <= 0.0:
        return vol, padded_init_ijk, gt_verts_dict_list

    B, C, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    mask = (torch.rand(B, device=device) < prob)
    if mask.sum().item() == 0:
        return vol, padded_init_ijk, gt_verts_dict_list

    rot_deg = float(getattr(cfg.dataset, "aug_rot_range_deg", 0.0))
    scale_range = float(getattr(cfg.dataset, "aug_scale_range", 0.0))
    trans_mm = float(getattr(cfg.dataset, "aug_trans_range_mm", 0.0))

    trans_ndc_xyz = torch.zeros((B, 3), device=device, dtype=dtype)
    den_xyz = torch.tensor([W - 1, H - 1, D - 1], device=device, dtype=dtype).clamp(min=1.0)

    for i in range(B):
        vsize_xyz = voxel_sizes_xyz_from_affine(affines[i].to(device=device, dtype=dtype))
        trans_vox_xyz = (trans_mm / vsize_xyz)
        trans_ndc_xyz[i] = 2.0 * (trans_vox_xyz / den_xyz)

    A_fwd, b_fwd = random_affine_ndc_xyz(B, rot_deg, scale_range, trans_ndc_xyz, device, dtype)

    I = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(B, 1, 1)
    Z = torch.zeros((B, 3), device=device, dtype=dtype)
    A_fwd = torch.where(mask.view(B, 1, 1), A_fwd, I)
    b_fwd = torch.where(mask.view(B, 1), b_fwd, Z)

    A_inv = torch.linalg.inv(A_fwd)
    b_inv = -(A_inv @ b_fwd.unsqueeze(-1)).squeeze(-1)

    theta = torch.zeros((B, 3, 4), device=device, dtype=dtype)
    theta[:, :, :3] = A_inv
    theta[:, :, 3] = b_inv

    grid = F.affine_grid(theta, size=vol.size(), align_corners=True)
    vol = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)

    for i in range(B):
        if not mask[i].item():
            continue

        L = int(lengths[i].item())

        v_ijk = padded_init_ijk[i, :L]
        v_xyz = ijk_to_xyz(v_ijk)
        u = voxel_to_ndc_xyz(v_xyz, D, H, W)
        u2 = (A_fwd[i] @ u.t()).t() + b_fwd[i].view(1, 3)
        v_xyz2 = ndc_to_voxel_xyz(u2, D, H, W)
        padded_init_ijk[i, :L] = xyz_to_ijk(v_xyz2)

        gdict = gt_verts_dict_list[i]
        for s in surface_names:
            gv_ijk = gdict[s]
            gv_xyz = ijk_to_xyz(gv_ijk)
            ug = voxel_to_ndc_xyz(gv_xyz, D, H, W)
            ug2 = (A_fwd[i] @ ug.t()).t() + b_fwd[i].view(1, 3)
            gv_xyz2 = ndc_to_voxel_xyz(ug2, D, H, W)
            gdict[s] = xyz_to_ijk(gv_xyz2)
        gt_verts_dict_list[i] = gdict

    return vol, padded_init_ijk, gt_verts_dict_list


def apply_intensity_aug(vol, cfg):
    """
    MRI-appearance augmentation applied ONLY to the MRI channel (vol[:, 0:1]).
    The geometry/probability channel(s) (vol[:, 1:]) are left untouched.

    All operations are safe on z-score-normalized MRI (values ~N(0,1), may be
    negative). This is the main regularizer for closing the train/val gap, since
    the affine aug in apply_aug() never perturbs intensity/appearance.

    Config (under cfg.dataset, all default to 0 = disabled):
      aug_intensity_prob : per-sample probability of applying intensity aug
      aug_bias_strength  : std of the smooth multiplicative bias field (e.g. 0.3)
      aug_gain_range     : +/- multiplicative contrast gain (e.g. 0.1 -> x*[0.9,1.1])
      aug_bright_range   : +/- additive brightness shift in z-units (e.g. 0.1)
      aug_noise_std      : std of additive Gaussian noise in z-units (e.g. 0.05)
    """
    prob = float(getattr(cfg.dataset, "aug_intensity_prob", 0.0))
    if prob <= 0.0:
        return vol

    B, C, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    mask = (torch.rand(B, device=device) < prob)
    if mask.sum().item() == 0:
        return vol

    bias_strength = float(getattr(cfg.dataset, "aug_bias_strength", 0.0))
    gain_range    = float(getattr(cfg.dataset, "aug_gain_range", 0.0))
    bright_range  = float(getattr(cfg.dataset, "aug_bright_range", 0.0))
    noise_std     = float(getattr(cfg.dataset, "aug_noise_std", 0.0))

    mri = vol[:, 0:1].clone()  # (B,1,D,H,W)

    for i in range(B):
        if not mask[i].item():
            continue

        x = mri[i:i + 1]  # (1,1,D,H,W)

        # 1) smooth multiplicative bias field (low-frequency -> upsampled -> exp)
        if bias_strength > 0.0:
            lo = torch.randn(1, 1, 4, 5, 4, device=device, dtype=dtype) * bias_strength
            field = F.interpolate(lo, size=(D, H, W), mode="trilinear", align_corners=True)
            x = x * torch.exp(field)

        # 2) global contrast gain
        if gain_range > 0.0:
            g = 1.0 + (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * gain_range
            x = x * g

        # 3) global brightness shift
        if bright_range > 0.0:
            b = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * bright_range
            x = x + b

        # 4) additive Gaussian noise
        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std

        mri[i:i + 1] = x

    if C > 1:
        vol = torch.cat([mri, vol[:, 1:]], dim=1)
    else:
        vol = mri

    return vol


# -----------------------
# Utilities for building padded init verts
# -----------------------
def build_merged_init_and_metadata(batch, device, surface_names):
    B = len(batch["init_verts_vox"])

    per_counts_init: List[List[int]] = []
    merged_init_list: List[torch.Tensor] = []
    init_faces_list: List[Dict[str, torch.Tensor]] = []
    gt_verts_list: List[Dict[str, torch.Tensor]] = []
    gt_faces_list: List[Dict[str, torch.Tensor]] = []

    for i in range(B):
        counts = []
        v_all = []
        f_init_dict = {}
        gv_dict = {}
        gf_dict = {}

        for s in surface_names:
            v = batch["init_verts_vox"][i][s].to(device)
            f = batch["init_faces"][i][s].to(device).long()
            gv = batch["gt_verts_vox"][i][s].to(device)
            gf = batch["gt_faces"][i][s].to(device).long()

            counts.append(int(v.shape[0]))
            v_all.append(v)
            f_init_dict[s] = f
            gv_dict[s] = gv
            gf_dict[s] = gf

        per_counts_init.append(counts)
        merged_init_list.append(torch.cat(v_all, dim=0))
        init_faces_list.append(f_init_dict)
        gt_verts_list.append(gv_dict)
        gt_faces_list.append(gf_dict)

    lengths = torch.tensor([v.shape[0] for v in merged_init_list], device=device, dtype=torch.long)
    Vmax = int(lengths.max().item())

    padded_init = torch.zeros((B, Vmax, 3), device=device, dtype=merged_init_list[0].dtype)
    for i in range(B):
        padded_init[i, :lengths[i]] = merged_init_list[i]

    return lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list


def save_model_state(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    _atomic_torch_save(net.state_dict(), path)


def extract_model_state_dict(checkpoint):
    """Return a model state_dict from either a raw state_dict or a full checkpoint."""
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state_dict-like object: {type(state)}")

    # Be tolerant of checkpoints saved from DataParallel/DDP wrappers.
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def move_optimizer_state_to_device(optimizer, device):
    """Move optimizer state tensors after loading a CPU checkpoint for CUDA training."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def build_adamw_param_groups(model, weight_decay: float):
    """
    AdamW parameter groups for 3D CNNs:
      - decay: convolution / projection weights
      - no_decay: biases, GroupNorm/norm parameters, and scalar/1D parameters
    This avoids applying weight decay to GroupNorm affine parameters and gates.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if name.endswith(".bias") or param.ndim <= 1 or ".gn" in lname or "norm" in lname:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _rng_state_for_checkpoint():
    state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    return state


def _restore_rng_state_from_checkpoint(ckpt, *, rank: int, device, cfg):
    """
    Restore RNG state from a rank-0/single-process checkpoint.

    In DDP this checkpoint is written only by rank 0, so exact per-rank RNG replay is
    not possible from this file alone. Rank 0 restores exactly; other ranks are
    reseeded deterministically with a rank/epoch offset to preserve augmentation
    diversity instead of cloning rank-0 random streams.
    """
    rng = ckpt.get("rng_state", None)
    if rng is None:
        return

    if rank == 0:
        torch_state = rng.get("torch", None)
        if torch_state is not None:
            torch.set_rng_state(torch_state)

        cuda_state = rng.get("cuda", None)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        numpy_state = rng.get("numpy", None)
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        random_state = rng.get("random", None)
        if random_state is not None:
            random.setstate(random_state)
    else:
        # Avoid identical augmentation streams on nonzero DDP ranks after resume.
        epoch_offset = int(ckpt.get("epoch", 0)) * 100000
        seed_all(int(cfg.trainer.seed) + epoch_offset, rank=rank)


def save_full_checkpoint(
    model,
    optimizer,
    scheduler,
    path: str,
    epoch: int,
    best_score: float,
    best_rmse_seen: float,
    best_model_epoch: int,
    best_rmse_epoch: int,
    no_improve: int,
    no_improve_rmse: int,
    cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "best_rmse_seen": best_rmse_seen,
        "best_model_epoch": best_model_epoch,
        "best_rmse_epoch": best_rmse_epoch,
        "no_improve": int(no_improve),
        "no_improve_rmse": int(no_improve_rmse),
        "rng_state": _rng_state_for_checkpoint(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    _atomic_torch_save(ckpt, path)


def fmt_collision_stats(total, hit, csum):
    if total <= 0:
        return "NA"
    pct = 100.0 * (hit / total)
    mean_all = csum / total
    mean_hit = csum / max(hit, 1.0)
    return f"{hit:.0f}/{total:.0f} ({pct:.2f}%) | MeanContacts(all)={mean_all:.2f} | MeanContacts(hit)={mean_hit:.2f}"


def compute_collision_percentages(
    lh_total: float,
    lh_hit: float,
    rh_total: float,
    rh_hit: float,
    lr_total: float,
    lr_hit: float,
) -> Tuple[float, float]:
    wp_total = lh_total + rh_total
    wp_hit = lh_hit + rh_hit
    wp_pct = 100.0 * wp_hit / max(wp_total, 1.0)
    lr_pct = 100.0 * lr_hit / max(lr_total, 1.0)
    return float(wp_pct), float(lr_pct)


# -----------------------
# Main
# -----------------------
@hydra.main(version_base=None, config_path="pkg://simcortex.configs.deform", config_name="train")
def main(cfg: DictConfig):
    rank, world_size, local_rank, is_distributed = setup_ddp()
    tb_writer = None
    file_handler = None

    try:
        user_config = str(OmegaConf.select(cfg, "user_config", default="") or "").strip()
        if user_config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(to_absolute_path(user_config)))

        validate_deform_training_config(cfg)

        level = getattr(logging, str(getattr(cfg.trainer, "log_level", "INFO")).upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            log.info("world_size=%d, local_rank=%d", world_size, local_rank)
            print(OmegaConf.to_yaml(cfg))

        seed_all(int(cfg.trainer.seed), rank=rank)
        torch.backends.cudnn.benchmark = True

        surface_names = [str(name).strip() for name in cfg.dataset.surface_name]
        inshape = tuple(int(x) for x in cfg.model.inshape)

        split_file = _as_abs_path(cfg.dataset.split_file, field="dataset.split_file")
        train_split = str(getattr(cfg.dataset, "train_split_name", "train")).strip()
        val_split = str(getattr(cfg.dataset, "val_split_name", "val")).strip()
        if not train_split or not val_split or train_split == val_split:
            raise ValueError(
                f"Training and validation split names must be non-empty and distinct, got "
                f"{train_split!r} and {val_split!r}"
            )

        session_label = str(getattr(cfg.dataset, "session_label", "01")).strip()
        space = str(getattr(cfg.dataset, "space", "MNI152")).strip()

        use_probability_map = bool(
            OmegaConf.select(
                cfg,
                "dataset.use_probability_map",
                default=True,
            )
        )
        use_fixed_initial_surface = bool(
            OmegaConf.select(
                cfg,
                "dataset.use_fixed_initial_surface",
                default=False,
            )
        )
        fixed_template_root = _optional_abs_path(
            OmegaConf.select(
                cfg,
                "dataset.fixed_template_root",
                default=None,
            )
        )
        needs_subject_initsurf = _requires_subject_initsurf(cfg)

        mode, single_preproc_root, single_initsurf_root, roots_map, initsurf_roots_map = _detect_deform_train_mode(cfg)
        if mode == "single":
            single_preproc_root = _as_abs_path(
                single_preproc_root,
                field="dataset.path",
            )
            if single_initsurf_root is not None:
                single_initsurf_root = _as_abs_path(
                    single_initsurf_root,
                    field="dataset.initsurf_root",
                )
        else:
            roots_map = {
                key: _as_abs_path(
                    value,
                    field=f"dataset.roots.{key}",
                )
                for key, value in roots_map.items()
            }

            if initsurf_roots_map is not None:
                initsurf_roots_map = {
                    key: _as_abs_path(
                        value,
                        field=f"dataset.initsurf_roots.{key}",
                    )
                    for key, value in initsurf_roots_map.items()
                }

        out_root = _as_abs_path(
            getattr(cfg.outputs, "root", getattr(cfg.outputs, "output_dir", "")),
            field="outputs.root",
        )
        resume_from = _optional_abs_path(getattr(cfg.trainer, "resume_from", ""))
        init_ckpt = _optional_abs_path(getattr(cfg.model, "init_ckpt", ""))
        allow_existing = bool(getattr(cfg.outputs, "allow_existing", False))
        _validate_output_root(
            out_root,
            resume_from=resume_from,
            allow_existing=allow_existing,
        )
        if resume_from and not os.path.isfile(resume_from):
            raise FileNotFoundError(f"trainer.resume_from does not exist: {resume_from}")
        if init_ckpt and not os.path.isfile(init_ckpt):
            raise FileNotFoundError(f"model.init_ckpt does not exist: {init_ckpt}")

        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"dataset.split_file does not exist: {split_file}")
        df = validate_split_dataframe(pd.read_csv(split_file), mode=mode)

        # ---- Multi-dataset mode ----
        if mode == "multi":
            train_sets = []
            val_sets = []

            for ds_key, ds_df in df.groupby("dataset"):
                if ds_key not in roots_map:
                    raise KeyError(
                        f"dataset.roots is missing key: {ds_key}"
                    )

                preproc_root = str(roots_map[ds_key])

                if initsurf_roots_map is None:
                    initsurf_root = None
                else:
                    initsurf_root = initsurf_roots_map.get(
                        ds_key,
                        None,
                    )

                if needs_subject_initsurf and initsurf_root is None:
                    raise KeyError(
                        "dataset.initsurf_roots is missing key "
                        f"required by the selected input mode: {ds_key}"
                    )

                tr_subs = ds_df[ds_df["split"].astype(str).str.strip() == train_split]["subject"].astype(str).tolist()
                va_subs = ds_df[ds_df["split"].astype(str).str.strip() == val_split]["subject"].astype(str).tolist()

                if len(tr_subs) > 0:
                    train_sets.append(
                        CSRDeformDataset(
                            preproc_root=preproc_root,
                            initsurf_root=initsurf_root,
                            subjects=tr_subs,
                            session_label=session_label,
                            space=space,
                            surface_names=surface_names,
                            inshape_dhw=inshape,
                            prob_clip_min=cfg.dataset.prob_clip_min,
                            prob_clip_max=cfg.dataset.prob_clip_max,
                            prob_gamma=cfg.dataset.prob_gamma,
                            strict_missing=bool(getattr(cfg.dataset, "strict_missing", True)),
                            use_probability_map=use_probability_map,
                            use_fixed_initial_surface=use_fixed_initial_surface,
                            fixed_template_root=fixed_template_root or None,
                        )
                    )

                if len(va_subs) > 0:
                    val_sets.append(
                        CSRDeformDataset(
                            preproc_root=preproc_root,
                            initsurf_root=initsurf_root,
                            subjects=va_subs,
                            session_label=session_label,
                            space=space,
                            surface_names=surface_names,
                            inshape_dhw=inshape,
                            prob_clip_min=cfg.dataset.prob_clip_min,
                            prob_clip_max=cfg.dataset.prob_clip_max,
                            prob_gamma=cfg.dataset.prob_gamma,
                            strict_missing=bool(getattr(cfg.dataset, "strict_missing", True)),
                            use_probability_map=use_probability_map,
                            use_fixed_initial_surface=use_fixed_initial_surface,
                            fixed_template_root=fixed_template_root or None,
                        )
                    )

            if len(train_sets) == 0:
                raise RuntimeError("No training subjects found (multi-dataset). Check split_file and train_split_name.")
            if len(val_sets) == 0:
                raise RuntimeError("No validation subjects found (multi-dataset). Check split_file and val_split_name.")

            train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
            val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

        # ---- Single-dataset mode ----
        else:
            tr_subs = df[df["split"].astype(str).str.strip() == train_split]["subject"].astype(str).tolist()
            va_subs = df[df["split"].astype(str).str.strip() == val_split]["subject"].astype(str).tolist()

            if len(tr_subs) == 0:
                raise RuntimeError("No training subjects found (single-dataset). Check split_file and train_split_name.")
            if len(va_subs) == 0:
                raise RuntimeError("No validation subjects found (single-dataset). Check split_file and val_split_name.")

            train_ds = CSRDeformDataset(
                preproc_root=str(single_preproc_root),
                initsurf_root=single_initsurf_root,
                subjects=tr_subs,
                session_label=session_label,
                space=space,
                surface_names=surface_names,
                inshape_dhw=inshape,
                prob_clip_min=cfg.dataset.prob_clip_min,
                prob_clip_max=cfg.dataset.prob_clip_max,
                prob_gamma=cfg.dataset.prob_gamma,
                strict_missing=bool(getattr(cfg.dataset, "strict_missing", True)),
                use_probability_map=use_probability_map,
                use_fixed_initial_surface=use_fixed_initial_surface,
                fixed_template_root=fixed_template_root or None,
            )

            val_ds = CSRDeformDataset(
                preproc_root=str(single_preproc_root),
                initsurf_root=single_initsurf_root,
                subjects=va_subs,
                session_label=session_label,
                space=space,
                surface_names=surface_names,
                inshape_dhw=inshape,
                prob_clip_min=cfg.dataset.prob_clip_min,
                prob_clip_max=cfg.dataset.prob_clip_max,
                prob_gamma=cfg.dataset.prob_gamma,
                strict_missing=bool(getattr(cfg.dataset, "strict_missing", True)),
                use_probability_map=use_probability_map,
                use_fixed_initial_surface=use_fixed_initial_surface,
                fixed_template_root=fixed_template_root or None,
            )

        _rank0_preflight_or_raise(
            train_ds,
            val_ds,
            rank=rank,
            is_distributed=is_distributed,
            surface_names=surface_names,
        )
        if rank == 0:
            log.info(
                "Deformation dataset preflight passed: train=%d validation=%d",
                len(train_ds),
                len(val_ds),
            )

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

        num_workers = int(cfg.trainer.num_workers)
        loader_common = dict(
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_csr_deform,
        )
        if num_workers > 0:
            loader_common.update(
                persistent_workers=bool(getattr(cfg.trainer, "persistent_workers", True)),
                prefetch_factor=int(getattr(cfg.trainer, "prefetch_factor", 2)),
            )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=int(cfg.trainer.img_batch_size),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            **loader_common,
        )

        # IMPORTANT: validation loader is NOT distributed to avoid sampler padding (77 -> 78)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(cfg.trainer.img_batch_size),
            shuffle=False,
            **loader_common,
        )

        if rank == 0:
            log.info("Loaded %d training subjects", len(train_ds))
            log.info("Loaded %d validation subjects", len(val_ds))

        # model
        if use_probability_map:
            raise ValueError(
                "MRI-only training requires "
                "dataset.use_probability_map=false."
            )

        if int(cfg.model.c_in) != 1:
            raise ValueError(
                "MRI-only training requires model.c_in=1, "
                f"but got {cfg.model.c_in}."
            )

        model = SurfDeform(
            C_hid=cfg.model.c_hid,
            C_in=int(cfg.model.c_in),
            inshape=inshape,
            sigma=float(cfg.model.sigma),
            gn_groups=int(getattr(cfg.model, "gn_groups", 8)),
            dropout=float(getattr(cfg.model, "dropout", 0.0)),
        ).to(device)

        # optional initialization checkpoint: model-only or full checkpoint are both supported.
        if init_ckpt:
            if rank == 0:
                log.info("Loading init_ckpt: %s", init_ckpt)
            raw_ckpt = _load_trusted_checkpoint(init_ckpt, map_location="cpu")
            sd = extract_model_state_dict(raw_ckpt)
            missing, unexpected = model.load_state_dict(
                sd,
                strict=bool(getattr(cfg.model, "init_strict", True)),
            )
            if rank == 0:
                log.info("Init load done. missing=%d unexpected=%d", len(missing), len(unexpected))

        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # optim
        optimizer = torch.optim.AdamW(
            build_adamw_param_groups(model, float(cfg.trainer.weight_decay)),
            lr=float(cfg.trainer.learning_rate),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.trainer.scheduler_factor),
            patience=int(cfg.trainer.scheduler_patience),
            threshold=float(cfg.trainer.scheduler_threshold_mm),
            threshold_mode=str(cfg.trainer.scheduler_threshold_mode),
            cooldown=int(cfg.trainer.scheduler_cooldown),
            min_lr=float(cfg.trainer.scheduler_min_lr),
        )

        # Logging & Config Saving
        if rank == 0:
            os.makedirs(out_root, exist_ok=True)

            log_path = os.path.join(out_root, "train.log")
            root_logger = logging.getLogger()
            root_logger.setLevel(level)

            for handler in list(root_logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)
                    handler.flush()
                    handler.close()

            log_mode = "a" if resume_from else "w"
            file_handler = logging.FileHandler(log_path, mode=log_mode, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            root_logger.addHandler(file_handler)

            tb_dir = os.path.join(out_root, "tb_logs")
            os.makedirs(tb_dir, exist_ok=True)

            log.info("TensorBoard logging to %s", tb_dir)
            log.info("Log file writing to %s (mode=%s)", log_path, log_mode)

            resolved_conf_yaml = OmegaConf.to_yaml(cfg, resolve=True)
            config_name = "config_resolved_resume.yaml" if resume_from else "config_resolved.yaml"
            config_path = os.path.join(out_root, config_name)
            config_bytes = resolved_conf_yaml.encode("utf-8")
            fd, temp_config = tempfile.mkstemp(
                prefix=f".{config_name}.", suffix=".tmp", dir=out_root
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(config_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_config, config_path)
            except Exception:
                try:
                    os.remove(temp_config)
                except FileNotFoundError:
                    pass
                raise
            log.info("Resolved config saved to %s", config_path)

            file_handler.flush()
            tb_writer = SummaryWriter(tb_dir)
            formatted_config = resolved_conf_yaml.replace("\n", "  \n")
            tb_writer.add_text(
                "Hyperparameters",
                f"### Training Configuration\n```yaml\n{formatted_config}\n```",
                0,
            )

        # weights
        chamfer_w = float(cfg.objective.chamfer_weight)
        chamfer_scale = float(getattr(cfg.objective, "chamfer_scale", 1.0))
        edge_w_base = float(cfg.objective.edge_loss_weight)
        normal_w_base = float(cfg.objective.normal_weight)
        reg_warmup = int(getattr(cfg.objective, "reg_warmup_epochs", 0))

        # separation weights/settings: white vs pial per hemisphere
        hd_w_base = float(getattr(cfg.objective, "hd_weight", 0.0))
        hd_p = float(getattr(cfg.objective, "hd_p", 0.05))
        hd_lam = float(getattr(cfg.objective, "hd_lambda_mm", 0.5))
        Phd = int(getattr(cfg.objective, "hd_points", 30000))

        # Pial-LR separation: lh_pial vs rh_pial
        pial_lr_w_base = float(getattr(cfg.objective, "pial_lr_hd_weight", 0.0))
        pial_lr_p = float(getattr(cfg.objective, "pial_lr_hd_p", hd_p))
        pial_lr_lam = float(getattr(cfg.objective, "pial_lr_hd_lambda_mm", hd_lam))
        pial_lr_pts = int(getattr(cfg.objective, "pial_lr_hd_points", Phd))

        # Signed nested white-pial loss
        signed_w_base = float(getattr(cfg.objective, "signed_nested_weight", 0.0))
        signed_margin = float(getattr(cfg.objective, "signed_margin_mm", 0.5))
        signed_points = int(getattr(cfg.objective, "signed_points", 40000))

        # train setup
        num_epochs = int(cfg.trainer.num_epochs)
        accum_steps = max(1, int(cfg.trainer.grad_accum_steps))
        grad_clip = float(cfg.trainer.grad_clip_norm)
        mesh_chunk = max(1, int(cfg.trainer.mesh_chunk))
        Ptrain = int(cfg.trainer.points_per_image)
        Pval = int(cfg.trainer.val_points_per_image)
        val_interval = max(1, int(cfg.trainer.validation_interval))
        col_interval = int(getattr(cfg.trainer, "collision_interval", val_interval))

        # Collision-aware checkpoint settings.
        # deform_best_model.pth is the final recommended model.
        alpha_wp = float(OmegaConf.select(cfg, "checkpoint.alpha_wp", default=0.006))
        alpha_lr = float(OmegaConf.select(cfg, "checkpoint.alpha_lr", default=0.002))
        rmse_guardrail_rel = float(OmegaConf.select(cfg, "checkpoint.rmse_guardrail_rel", default=1.06))
        score_delta = float(OmegaConf.select(cfg, "checkpoint.min_delta_score", default=1e-4))
        require_collision_for_best = bool(OmegaConf.select(cfg, "checkpoint.require_collision_for_best", default=True))

        if rank == 0:
            log.info(
                "Collision-aware checkpointing: alpha_wp=%.4f alpha_lr=%.4f "
                "rmse_guardrail_rel=%.4f min_delta_score=%.6f require_collision_for_best=%s",
                alpha_wp, alpha_lr, rmse_guardrail_rel, score_delta, require_collision_for_best,
            )
            if col_interval != val_interval:
                log.warning(
                    "collision_interval (%d) != validation_interval (%d). "
                    "Collision-aware model selection can only update on epochs with collision metrics. "
                    "Recommended: set collision_interval == validation_interval.",
                    col_interval, val_interval,
                )

        if require_collision_for_best and not HAS_FCL:
            raise RuntimeError(
                "checkpoint.require_collision_for_best=True but the trimesh/python-fcl "
                "collision backend is unavailable. Install python-fcl or set "
                "checkpoint.require_collision_for_best=False."
            )

        # Diagnostic best RMSE checkpoint.
        best_rmse_seen = float("inf")
        best_rmse_epoch = -1

        # Final model selection checkpoint.
        best_score = float("inf")
        best_model_epoch = -1

        no_improve = 0
        no_improve_rmse = 0
        early_patience = int(getattr(cfg.trainer, "early_stop_patience", 0))
        # RMSE delta is used only for diagnostic best-rmse checkpoint.
        early_rmse_delta = float(getattr(cfg.trainer, "early_stop_min_delta_mm", 0.0))

        start_epoch = 1
        if resume_from:
            if rank == 0:
                log.info("Resuming from full checkpoint: %s", resume_from)
            ckpt = _load_trusted_checkpoint(resume_from, map_location="cpu")
            if not isinstance(ckpt, dict) or "model" not in ckpt or "optimizer" not in ckpt:
                raise ValueError(
                    "trainer.resume_from must point to a full checkpoint containing at least "
                    "'model' and 'optimizer'. Use model.init_ckpt for model-only initialization."
                )

            net = model.module if hasattr(model, "module") else model
            net.load_state_dict(extract_model_state_dict(ckpt), strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_state_to_device(optimizer, device)

            if scheduler is not None and ckpt.get("scheduler", None) is not None:
                scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score = float(ckpt.get("best_score", best_score))
            best_rmse_seen = float(ckpt.get("best_rmse_seen", best_rmse_seen))
            best_model_epoch = int(ckpt.get("best_model_epoch", best_model_epoch))
            best_rmse_epoch = int(ckpt.get("best_rmse_epoch", best_rmse_epoch))
            no_improve = int(ckpt.get("no_improve", no_improve))
            no_improve_rmse = int(ckpt.get("no_improve_rmse", no_improve_rmse))
            _restore_rng_state_from_checkpoint(ckpt, rank=rank, device=device, cfg=cfg)

            if rank == 0:
                log.info(
                    "Resume state loaded: start_epoch=%d best_score=%.6f best_rmse=%.6f "
                    "best_model_epoch=%d best_rmse_epoch=%d no_improve=%d no_improve_rmse=%d",
                    start_epoch, best_score, best_rmse_seen, best_model_epoch, best_rmse_epoch,
                    no_improve, no_improve_rmse,
                )

        # -----------------------
        # Training loop
        # -----------------------
        for epoch in range(start_epoch, num_epochs + 1):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if rank == 0:
                log.info("Epoch %d/%d", epoch, num_epochs)

            # warmup for regularizers (including separation penalties)
            t = 1.0
            if reg_warmup > 0:
                t = min(1.0, epoch / float(reg_warmup))
            edge_w = edge_w_base * t
            normal_w = normal_w_base * t
            hd_w_eff = hd_w_base * t
            pial_lr_w_eff = pial_lr_w_base * t
            signed_w_eff = signed_w_base * t

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # epoch stats (sum over meshes)
            chamfer_sq_sum = 0.0
            edge_sum = 0.0
            normal_sum = 0.0
            mesh_count = 0.0

            total_obj_sum = 0.0
            total_obj_count = 0.0

            # separation stats (sum over pairs)
            sep_pen_sum = 0.0
            sep_q_sum = 0.0
            sep_count = 0.0

            # Pial-LR stats (sum over pairs)
            pial_lr_pen_sum = 0.0
            pial_lr_sep_q_sum = 0.0
            pial_lr_count = 0.0

            # Signed nested stats
            signed_pen_sum = 0.0
            signed_badw_sum = 0.0
            signed_badp_sum = 0.0
            signed_wmean_sum = 0.0
            signed_pmean_sum = 0.0
            signed_count = 0.0



            surf_stats = {s: {"chamfer_sq": 0.0, "count": 0.0} for s in surface_names}

            num_train_batches = len(train_loader)

            for batch_idx, batch in enumerate(tqdm(train_loader, disable=(rank != 0), desc=f"Train {epoch} [r{rank}]")):
                window_start = (batch_idx // accum_steps) * accum_steps
                current_accum_size = min(accum_steps, num_train_batches - window_start)
                is_last_micro_in_window = ((batch_idx + 1) == num_train_batches) or (((batch_idx + 1) % accum_steps) == 0)

                vol = batch["vol"].to(device, non_blocking=True)
                aff = batch["affine"].to(device, non_blocking=True)
                shift = batch["shift_ijk"].to(device, non_blocking=True)

                B, _, D, H, W = vol.shape

                lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list = \
                    build_merged_init_and_metadata(batch, device, surface_names)

                # augmentation
                vol, padded_init, gt_verts_list = apply_aug(
                    vol=vol,
                    padded_init_ijk=padded_init,
                    lengths=lengths,
                    gt_verts_dict_list=gt_verts_list,
                    affines=aff,
                    cfg=cfg,
                    surface_names=surface_names,
                )

                # MRI-appearance augmentation (intensity/bias/noise), MRI channel only
                vol = apply_intensity_aug(vol, cfg)


                pred_vox = model(
                    padded_init,
                    vol,
                    int(cfg.model.n_steps),
                )

                # Build mesh lists in WORLD(mm) for Chamfer/edge/normal
                pred_verts_mm, pred_faces = [], []
                init_verts_mm = []
                gt_verts_mm, gt_faces = [], []
                surf_of_mesh = []

                # store pred meshes per sample for separation losses
                pred_mesh_mm_per_sample = [dict() for _ in range(B)]
                invalid_mesh_messages: List[str] = []

                for i in range(B):
                    pred_i = pred_vox[i, :lengths[i]]
                    init_i = padded_init[i, :lengths[i]]
                    splits = torch.split(pred_i, per_counts_init[i], dim=0)
                    init_splits = torch.split(init_i, per_counts_init[i], dim=0)

                    A = aff[i]
                    sh = shift[i].view(1, 3)

                    for j, s in enumerate(surface_names):
                        pv = splits[j]
                        iv = init_splits[j]
                        gv = gt_verts_list[i][s]

                        f = init_faces_list[i][s]
                        gf = gt_faces_list[i][s]

                        pv_mm = voxel_to_world(pv - sh, A)
                        iv_mm = voxel_to_world(iv - sh, A)
                        gv_mm = voxel_to_world(gv - sh, A)

                        pred_valid = mesh_is_valid(pv_mm, f)
                        init_valid = mesh_is_valid(iv_mm, f)
                        gt_valid = mesh_is_valid(gv_mm, gf)
                        if not (pred_valid and init_valid and gt_valid):
                            subject = batch["subject"][i]
                            invalid_mesh_messages.append(
                                f"{subject}/{s}: pred_valid={pred_valid}, "
                                f"init_valid={init_valid}, gt_valid={gt_valid}"
                            )
                            continue

                        pred_mesh_mm_per_sample[i][s] = (pv_mm, f)
                        pred_verts_mm.append(pv_mm)
                        pred_faces.append(f)
                        init_verts_mm.append(iv_mm)
                        gt_verts_mm.append(gv_mm)
                        gt_faces.append(gf)
                        surf_of_mesh.append(s)

                M = len(pred_verts_mm)
                expected_meshes = B * len(surface_names)
                if M != expected_meshes:
                    invalid_mesh_messages.append(
                        f"batch coverage={M}/{expected_meshes} valid surfaces"
                    )
                _raise_if_any_rank_error(
                    invalid_mesh_messages,
                    device=device,
                    is_distributed=is_distributed,
                    context=f"Invalid deformation mesh at epoch={epoch}, batch={batch_idx}",
                )

                # -----------------------
                # White-pial separation penalty
                # -----------------------
                loss_sep = torch.zeros((), device=device)
                pair_count = 0
                sep_q_sum_batch = 0.0

                if hd_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]

                        if ("lh_white" in md) and ("lh_pial" in md):
                            vw, fw = md["lh_white"]
                            vp, fp = md["lh_pial"]
                            mw = Meshes(verts=[vw], faces=[fw])
                            mpial = Meshes(verts=[vp], faces=[fp])
                            sep_q, pen = partial_hd_penalty(mw, mpial, p=hd_p, lam=hd_lam, n_pts=Phd)
                            loss_sep = loss_sep + pen
                            sep_q_sum_batch += float(sep_q.detach().item())
                            pair_count += 1

                        if ("rh_white" in md) and ("rh_pial" in md):
                            vw, fw = md["rh_white"]
                            vp, fp = md["rh_pial"]
                            mw = Meshes(verts=[vw], faces=[fw])
                            mpial = Meshes(verts=[vp], faces=[fp])
                            sep_q, pen = partial_hd_penalty(mw, mpial, p=hd_p, lam=hd_lam, n_pts=Phd)
                            loss_sep = loss_sep + pen
                            sep_q_sum_batch += float(sep_q.detach().item())
                            pair_count += 1

                    if pair_count > 0:
                        loss_sep = loss_sep / float(pair_count)

                # -----------------------
                # Signed nested white-pial penalty
                # -----------------------
                loss_signed = torch.zeros((), device=device)
                signed_pair_count = 0
                signed_badw_batch_sum = 0.0
                signed_badp_batch_sum = 0.0
                signed_wmean_batch_sum = 0.0
                signed_pmean_batch_sum = 0.0

                if signed_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]

                        if ("lh_white" in md) and ("lh_pial" in md):
                            vw, fw = md["lh_white"]
                            vp, fp = md["lh_pial"]

                            lsgn, badw, badp, meanw, meanp = signed_nested_surface_penalty(
                                vw, fw, vp, fp,
                                margin_mm=signed_margin,
                                n_points=signed_points,
                            )
                            loss_signed = loss_signed + lsgn
                            signed_badw_batch_sum += badw
                            signed_badp_batch_sum += badp
                            signed_wmean_batch_sum += meanw
                            signed_pmean_batch_sum += meanp
                            signed_pair_count += 1

                        if ("rh_white" in md) and ("rh_pial" in md):
                            vw, fw = md["rh_white"]
                            vp, fp = md["rh_pial"]

                            lsgn, badw, badp, meanw, meanp = signed_nested_surface_penalty(
                                vw, fw, vp, fp,
                                margin_mm=signed_margin,
                                n_points=signed_points,
                            )
                            loss_signed = loss_signed + lsgn
                            signed_badw_batch_sum += badw
                            signed_badp_batch_sum += badp
                            signed_wmean_batch_sum += meanw
                            signed_pmean_batch_sum += meanp
                            signed_pair_count += 1

                    if signed_pair_count > 0:
                        loss_signed = loss_signed / float(signed_pair_count)
                # -----------------------
                # Pial-LR separation: lh_pial vs rh_pial
                # -----------------------
                loss_pial_lr = torch.zeros((), device=device)
                pial_lr_pair_count = 0
                pial_lr_sep_q_sum_batch = 0.0

                if pial_lr_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]
                        if ("lh_pial" in md) and ("rh_pial" in md):
                            vl, fl = md["lh_pial"]
                            vr, fr = md["rh_pial"]
                            ml = Meshes(verts=[vl], faces=[fl])
                            mr = Meshes(verts=[vr], faces=[fr])

                            sep_q_lr, pen_lr = partial_hd_penalty(
                                ml, mr, p=pial_lr_p, lam=pial_lr_lam, n_pts=pial_lr_pts
                            )
                            loss_pial_lr = loss_pial_lr + pen_lr
                            pial_lr_sep_q_sum_batch += float(sep_q_lr.detach().item())
                            pial_lr_pair_count += 1

                    if pial_lr_pair_count > 0:
                        loss_pial_lr = loss_pial_lr / float(pial_lr_pair_count)

                # -----------------------
                # Chamfer/edge/normal losses (chunked)
                # -----------------------
                # chamfer_distance returns mean squared distances (mmÂ²).
                # Logs report sqrt(loss_chamfer_mm2) as ChamferRMSE in mm.
                loss_chamfer_mm2 = torch.zeros((), device=device)
                loss_edge = torch.zeros((), device=device)
                loss_norm = torch.zeros((), device=device)

                chamfer_sq_det_sum = 0.0

                for start in range(0, M, mesh_chunk):
                    end = min(M, start + mesh_chunk)

                    mpred = Meshes(verts=pred_verts_mm[start:end], faces=pred_faces[start:end])
                    mgt = Meshes(verts=gt_verts_mm[start:end], faces=gt_faces[start:end])

                    pp = sample_points_from_meshes(mpred, num_samples=Ptrain)
                    pg = sample_points_from_meshes(mgt, num_samples=Ptrain)

                    chamfer_sq_per, _ = chamfer_distance(pp, pg, batch_reduction=None)
                    n = mesh_normal_consistency(mpred)

                    mchunk = (end - start)

                    edge_chunk = torch.zeros((), device=device)
                    for k in range(mchunk):
                        edge_chunk = edge_chunk + edge_length_preservation_loss(
                            pred_verts_mm[start + k],
                            init_verts_mm[start + k],
                            pred_faces[start + k],
                        )
                    edge_chunk = edge_chunk / float(max(mchunk, 1))

                    loss_chamfer_mm2 = loss_chamfer_mm2 + chamfer_sq_per.mean() * mchunk
                    loss_edge = loss_edge + edge_chunk * mchunk
                    loss_norm = loss_norm + n * mchunk

                    chamfer_sq_det_sum += float(chamfer_sq_per.detach().sum().item())
                    for k in range(mchunk):
                        ss = surf_of_mesh[start + k]
                        surf_stats[ss]["chamfer_sq"] += float(chamfer_sq_per[k].detach().item())
                        surf_stats[ss]["count"] += 1.0

                loss_chamfer_mm2 = loss_chamfer_mm2 / M
                loss_edge = loss_edge / M
                loss_norm = loss_norm / M

                # total loss
                total_loss = (
                    chamfer_w * (chamfer_scale * loss_chamfer_mm2)
                    + edge_w * loss_edge
                    + normal_w * loss_norm
                    + hd_w_eff * loss_sep
                    + pial_lr_w_eff * loss_pial_lr
                    + signed_w_eff * loss_signed
                )

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        "Non-finite total_loss at "
                        f"epoch={epoch}, batch_idx={batch_idx}: "
                        f"loss_chamfer_mm2={float(loss_chamfer_mm2.detach().item()):.8g}, "
                        f"loss_edge={float(loss_edge.detach().item()):.8g}, "
                        f"loss_norm={float(loss_norm.detach().item()):.8g}, "
                        f"loss_sep={float(loss_sep.detach().item()):.8g}, "
                        f"loss_pial_lr={float(loss_pial_lr.detach().item()):.8g}, "
                        f"loss_signed={float(loss_signed.detach().item()):.8g}"
                    )

                loss_to_back = total_loss / float(current_accum_size)

                sync_ctx = nullcontext()
                if is_distributed and hasattr(model, "no_sync") and (not is_last_micro_in_window):
                    sync_ctx = model.no_sync()

                with sync_ctx:
                    loss_to_back.backward()

                if is_last_micro_in_window:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # stats
                chamfer_sq_sum += chamfer_sq_det_sum
                edge_sum += float((loss_edge.detach() * M).item())
                normal_sum += float((loss_norm.detach() * M).item())
                mesh_count += float(M)

                total_obj_sum += float(total_loss.detach().item())
                total_obj_count += 1.0

                if pair_count > 0:
                    sep_pen_sum += float((loss_sep.detach() * pair_count).item())
                    sep_q_sum += float(sep_q_sum_batch)
                    sep_count += float(pair_count)

                if pial_lr_pair_count > 0:
                    pial_lr_pen_sum += float((loss_pial_lr.detach() * pial_lr_pair_count).item())
                    pial_lr_sep_q_sum += float(pial_lr_sep_q_sum_batch)
                    pial_lr_count += float(pial_lr_pair_count)

                if signed_pair_count > 0:
                    signed_pen_sum += float((loss_signed.detach() * signed_pair_count).item())
                    signed_badw_sum += float(signed_badw_batch_sum)
                    signed_badp_sum += float(signed_badp_batch_sum)
                    signed_wmean_sum += float(signed_wmean_batch_sum)
                    signed_pmean_sum += float(signed_pmean_batch_sum)
                    signed_count += float(signed_pair_count)

            # reduce train stats
            if is_distributed:
                tstat = torch.tensor(
                    [
                        chamfer_sq_sum, edge_sum, normal_sum, mesh_count,
                        total_obj_sum, total_obj_count,
                        sep_pen_sum, sep_q_sum, sep_count,
                        pial_lr_pen_sum, pial_lr_sep_q_sum, pial_lr_count,
                        signed_pen_sum, signed_badw_sum, signed_badp_sum,
                        signed_wmean_sum, signed_pmean_sum, signed_count,
                    ],
                    device=device, dtype=torch.float64,
                )
                dist.all_reduce(tstat, op=dist.ReduceOp.SUM)

                (
                    chamfer_sq_sum, edge_sum, normal_sum, mesh_count,
                    total_obj_sum, total_obj_count,
                    sep_pen_sum, sep_q_sum, sep_count,
                    pial_lr_pen_sum, pial_lr_sep_q_sum, pial_lr_count,
                    signed_pen_sum, signed_badw_sum, signed_badp_sum,
                    signed_wmean_sum, signed_pmean_sum, signed_count,
                ) = tstat.tolist()


                surf_tensor = torch.zeros((len(surface_names), 2), device=device, dtype=torch.float64)
                for i, s in enumerate(surface_names):
                    surf_tensor[i, 0] = surf_stats[s]["chamfer_sq"]
                    surf_tensor[i, 1] = surf_stats[s]["count"]

                dist.all_reduce(surf_tensor, op=dist.ReduceOp.SUM)

                surf_global = {
                    s: {"chamfer_sq": surf_tensor[i, 0].item(), "count": surf_tensor[i, 1].item()}
                    for i, s in enumerate(surface_names)
                }
            else:
                surf_global = surf_stats

            # log train
            if rank == 0 and mesh_count > 0:
                chamfer_sq_mean = chamfer_sq_sum / mesh_count
                rmse_mm_train = math.sqrt(max(chamfer_sq_mean, 0.0))
                edge_mean = edge_sum / mesh_count
                norm_mean = normal_sum / mesh_count
                total_mean = total_obj_sum / max(total_obj_count, 1.0)

                if sep_count > 0:
                    sep_pen_mean = sep_pen_sum / sep_count
                    sep_q_mean_mm = sep_q_sum / sep_count
                else:
                    sep_pen_mean = 0.0
                    sep_q_mean_mm = 0.0

                if pial_lr_count > 0:
                    pial_lr_pen_mean = pial_lr_pen_sum / pial_lr_count
                    pial_lr_sep_q_mean = pial_lr_sep_q_sum / pial_lr_count
                else:
                    pial_lr_pen_mean = 0.0
                    pial_lr_sep_q_mean = 0.0

                if signed_count > 0:
                    signed_pen_mean = signed_pen_sum / signed_count
                    signed_badw_mean = signed_badw_sum / signed_count
                    signed_badp_mean = signed_badp_sum / signed_count
                    signed_wmean_mean = signed_wmean_sum / signed_count
                    signed_pmean_mean = signed_pmean_sum / signed_count
                else:
                    signed_pen_mean = 0.0
                    signed_badw_mean = 0.0
                    signed_badp_mean = 0.0
                    signed_wmean_mean = 0.0
                    signed_pmean_mean = 0.0


                surf_str = ", ".join(
                    f"{s}={math.sqrt(max(surf_global[s]['chamfer_sq']/max(surf_global[s]['count'],1.0),0.0)):.4f}mm"
                    for s in surface_names
                )

                log.info(
                    "Epoch %d [Train] | ChamferRMSE=%.4f mm | Edge=%.6f | Normal=%.6f | "
                    "SepPen=%.6f | SepQ=%.4f mm | wSep=%.4f | "
                    "PialLRSepPen=%.6f | PialLRSepQ=%.4f mm | wPialLR=%.4f | "
                    "SignedPen=%.6f | SignedBadW=%.2f%% | SignedBadP=%.2f%% | "
                    "SignedWMean=%.4f mm | SignedPMean=%.4f mm | wSigned=%.4f | "
                    "TotalObj=%.6f | Surfaces: %s",
                    epoch,
                    rmse_mm_train,
                    edge_mean,
                    norm_mean,
                    sep_pen_mean,
                    sep_q_mean_mm,
                    hd_w_eff,
                    pial_lr_pen_mean,
                    pial_lr_sep_q_mean,
                    pial_lr_w_eff,
                    signed_pen_mean,
                    signed_badw_mean,
                    signed_badp_mean,
                    signed_wmean_mean,
                    signed_pmean_mean,
                    signed_w_eff,
                    total_mean,
                    surf_str,
                )

                if tb_writer is not None:
                    net0 = model.module if hasattr(model, "module") else model
                    if hasattr(net0, "munet"):
                        for name, module in net0.munet.named_modules():
                            if hasattr(module, "gate_logit"):
                                gate_value = torch.sigmoid(module.gate_logit.detach()).item()
                                tb_writer.add_scalar(f"gates/{name}", gate_value, epoch)

                    tb_writer.add_scalar("train/rmse_mm", rmse_mm_train, epoch)
                    tb_writer.add_scalar("train/edge", edge_mean, epoch)
                    tb_writer.add_scalar("train/normal", norm_mean, epoch)
                    tb_writer.add_scalar("train/total_obj", total_mean, epoch)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

                    tb_writer.add_scalar("train/sep_penalty", sep_pen_mean, epoch)
                    tb_writer.add_scalar("train/sep_q_mean_mm", sep_q_mean_mm, epoch)
                    tb_writer.add_scalar("train/sep_weight_eff", hd_w_eff, epoch)

                    tb_writer.add_scalar("train/pial_lr_sep_penalty", pial_lr_pen_mean, epoch)
                    tb_writer.add_scalar("train/pial_lr_sep_q_mean_mm", pial_lr_sep_q_mean, epoch)
                    tb_writer.add_scalar("train/pial_lr_weight_eff", pial_lr_w_eff, epoch)

                    tb_writer.add_scalar("train/signed_penalty", signed_pen_mean, epoch)
                    tb_writer.add_scalar("train/signed_bad_white_pct", signed_badw_mean, epoch)
                    tb_writer.add_scalar("train/signed_bad_pial_pct", signed_badp_mean, epoch)
                    tb_writer.add_scalar("train/signed_white_mean_mm", signed_wmean_mean, epoch)
                    tb_writer.add_scalar("train/signed_pial_mean_mm", signed_pmean_mean, epoch)
                    tb_writer.add_scalar("train/signed_weight_eff", signed_w_eff, epoch)


            # -----------------------
            # Validation (rank 0 only) + complete collision coverage
            # -----------------------
            stop_tensor = torch.tensor(0, device=device, dtype=torch.int64)
            collision_error_tensor = torch.tensor(0, device=device, dtype=torch.int64)
            validation_error_tensor = torch.tensor(0, device=device, dtype=torch.int64)

            if (epoch % val_interval) == 0:
                # Use the underlying module to avoid DDP collectives in rank-0-only validation.
                net = model.module if hasattr(model, "module") else model
                net.eval()

                do_collision_check = (epoch % col_interval) == 0
                rmse_tensor = torch.tensor(float("inf"), device=device, dtype=torch.float64)
                score_tensor = torch.tensor(float("inf"), device=device, dtype=torch.float64)

                expected_val_subjects = len(val_ds)
                val_chamfer_sq_sum = 0.0
                val_count = 0.0
                val_surf = {
                    surface: {"chamfer_sq": 0.0, "count": 0.0}
                    for surface in surface_names
                }
                invalid_val_meshes: List[str] = []

                lh_total = lh_hit = lh_contacts_sum = 0.0
                rh_total = rh_hit = rh_contacts_sum = 0.0
                lr_total = lr_hit = lr_contacts_sum = 0.0
                lh_unknown = rh_unknown = lr_unknown = 0.0

                if rank == 0:
                    try:
                        with torch.no_grad():
                            for batch in tqdm(val_loader, disable=False, desc=f"Val {epoch} [rank0]"):
                                vol = batch["vol"].to(device, non_blocking=True)
                                aff = batch["affine"].to(device, non_blocking=True)
                                shift = batch["shift_ijk"].to(device, non_blocking=True)

                                B = int(vol.shape[0])
                                per_counts_init = []
                                merged_init_list = []
                                for i in range(B):
                                    verts = []
                                    counts = []
                                    for surface in surface_names:
                                        value = batch["init_verts_vox"][i][surface].to(device)
                                        verts.append(value)
                                        counts.append(int(value.shape[0]))
                                    per_counts_init.append(counts)
                                    merged_init_list.append(torch.cat(verts, dim=0))

                                lengths = torch.tensor(
                                    [value.shape[0] for value in merged_init_list],
                                    device=device,
                                    dtype=torch.long,
                                )
                                vmax = int(lengths.max().item())
                                padded_init = torch.zeros(
                                    (B, vmax, 3),
                                    device=device,
                                    dtype=merged_init_list[0].dtype,
                                )
                                for i in range(B):
                                    padded_init[i, : lengths[i]] = merged_init_list[i]

                                pred_vox = net(padded_init, vol, int(cfg.model.n_steps))

                                for i in range(B):
                                    subject = batch["subject"][i]
                                    affine = aff[i]
                                    subject_shift = shift[i].view(1, 3)

                                    pred_i = pred_vox[i, : lengths[i]]
                                    splits = torch.split(pred_i, per_counts_init[i], dim=0)

                                    pred_mm: Dict[str, torch.Tensor] = {}
                                    pred_f: Dict[str, torch.Tensor] = {}

                                    for j, surface in enumerate(surface_names):
                                        pred_vertices = splits[j]
                                        gt_vertices = batch["gt_verts_vox"][i][surface].to(device)

                                        pred_vertices_mm = voxel_to_world(
                                            pred_vertices - subject_shift,
                                            affine,
                                        )
                                        gt_vertices_mm = voxel_to_world(
                                            gt_vertices - subject_shift,
                                            affine,
                                        )

                                        faces = batch["init_faces"][i][surface].to(device).long()
                                        gt_faces = batch["gt_faces"][i][surface].to(device).long()

                                        pred_valid = mesh_is_valid(pred_vertices_mm, faces)
                                        gt_valid = mesh_is_valid(gt_vertices_mm, gt_faces)
                                        if not (pred_valid and gt_valid):
                                            invalid_val_meshes.append(
                                                f"{subject}/{surface}: pred_valid={pred_valid}, "
                                                f"gt_valid={gt_valid}"
                                            )
                                            continue

                                        pred_mm[surface] = pred_vertices_mm
                                        pred_f[surface] = faces

                                        pred_mesh = Meshes(verts=[pred_vertices_mm], faces=[faces])
                                        gt_mesh = Meshes(verts=[gt_vertices_mm], faces=[gt_faces])
                                        pred_points = sample_points_from_meshes(
                                            pred_mesh,
                                            num_samples=Pval,
                                        )
                                        gt_points = sample_points_from_meshes(
                                            gt_mesh,
                                            num_samples=Pval,
                                        )
                                        val_chamfer_sq, _ = chamfer_distance(
                                            pred_points,
                                            gt_points,
                                        )

                                        value = float(val_chamfer_sq.item())
                                        val_chamfer_sq_sum += value
                                        val_count += 1.0
                                        val_surf[surface]["chamfer_sq"] += value
                                        val_surf[surface]["count"] += 1.0

                                    if do_collision_check and HAS_FCL:
                                        if {"lh_white", "lh_pial"}.issubset(pred_mm):
                                            is_collision, contacts = count_collisions_inmemory(
                                                pred_mm["lh_white"], pred_f["lh_white"],
                                                pred_mm["lh_pial"], pred_f["lh_pial"],
                                            )
                                            if is_collision is None:
                                                lh_unknown += 1.0
                                            else:
                                                lh_total += 1.0
                                                lh_hit += 1.0 if is_collision else 0.0
                                                lh_contacts_sum += float(contacts)
                                        else:
                                            lh_unknown += 1.0

                                        if {"rh_white", "rh_pial"}.issubset(pred_mm):
                                            is_collision, contacts = count_collisions_inmemory(
                                                pred_mm["rh_white"], pred_f["rh_white"],
                                                pred_mm["rh_pial"], pred_f["rh_pial"],
                                            )
                                            if is_collision is None:
                                                rh_unknown += 1.0
                                            else:
                                                rh_total += 1.0
                                                rh_hit += 1.0 if is_collision else 0.0
                                                rh_contacts_sum += float(contacts)
                                        else:
                                            rh_unknown += 1.0

                                        if {"lh_pial", "rh_pial"}.issubset(pred_mm):
                                            is_collision, contacts = count_collisions_inmemory(
                                                pred_mm["lh_pial"], pred_f["lh_pial"],
                                                pred_mm["rh_pial"], pred_f["rh_pial"],
                                            )
                                            if is_collision is None:
                                                lr_unknown += 1.0
                                            else:
                                                lr_total += 1.0
                                                lr_hit += 1.0 if is_collision else 0.0
                                                lr_contacts_sum += float(contacts)
                                        else:
                                            lr_unknown += 1.0

                        validation_complete, validation_problem = validation_coverage_status(
                            val_count,
                            val_surf,
                            expected_subjects=expected_val_subjects,
                            surface_names=surface_names,
                        )
                        if invalid_val_meshes:
                            validation_complete = False
                            detail = "; ".join(invalid_val_meshes[:8])
                            validation_problem = ", ".join(
                                item for item in (validation_problem, detail) if item
                            )

                        if not validation_complete:
                            validation_error_tensor.fill_(1)
                            log.error(
                                "Epoch %d [Val] | Incomplete validation coverage: %s",
                                epoch,
                                validation_problem or "unknown validation error",
                            )
                        else:
                            chamfer_sq_mean = val_chamfer_sq_sum / val_count
                            rmse_mm = math.sqrt(max(chamfer_sq_mean, 0.0))
                            rmse_tensor.fill_(rmse_mm)

                            surface_summary = ", ".join(
                                f"{surface}="
                                f"{math.sqrt(max(val_surf[surface]['chamfer_sq'] / val_surf[surface]['count'], 0.0)):.4f}mm"
                                for surface in surface_names
                            )
                            log.info(
                                "Epoch %d [Val] | ChamferRMSE=%.4f mm | Surfaces: %s | "
                                "coverage=%d/%d",
                                epoch,
                                rmse_mm,
                                surface_summary,
                                int(val_count),
                                expected_val_subjects * len(surface_names),
                            )

                            collision_complete = False
                            collision_problem = ""
                            if do_collision_check and HAS_FCL:
                                collision_complete, collision_problem = collision_coverage_status(
                                    lh_total,
                                    rh_total,
                                    lr_total,
                                    lh_unknown,
                                    rh_unknown,
                                    lr_unknown,
                                    expected_subjects=expected_val_subjects,
                                )

                            collision_available = bool(
                                do_collision_check and HAS_FCL and collision_complete
                            )
                            wp_pct = 0.0
                            lr_pct = 0.0
                            score = rmse_mm

                            if do_collision_check:
                                if not HAS_FCL:
                                    log.error(
                                        "Epoch %d [Val] | Collision check unavailable: python-fcl is not working.",
                                        epoch,
                                    )
                                elif not collision_complete:
                                    log.error(
                                        "Epoch %d [Val] | Incomplete collision coverage: %s",
                                        epoch,
                                        collision_problem,
                                    )
                                else:
                                    log.info(
                                        "Epoch %d [Val] | White-Pial Collisions LH: %s",
                                        epoch,
                                        fmt_collision_stats(lh_total, lh_hit, lh_contacts_sum),
                                    )
                                    log.info(
                                        "Epoch %d [Val] | White-Pial Collisions RH: %s",
                                        epoch,
                                        fmt_collision_stats(rh_total, rh_hit, rh_contacts_sum),
                                    )
                                    log.info(
                                        "Epoch %d [Val] | Pial-Pial Collisions LR: %s",
                                        epoch,
                                        fmt_collision_stats(lr_total, lr_hit, lr_contacts_sum),
                                    )
                                    wp_pct, lr_pct = compute_collision_percentages(
                                        lh_total,
                                        lh_hit,
                                        rh_total,
                                        rh_hit,
                                        lr_total,
                                        lr_hit,
                                    )
                                    score = rmse_mm + alpha_wp * wp_pct + alpha_lr * lr_pct
                                    log.info(
                                        "Epoch %d [ValScore] | Score=%.4f | RMSE=%.4f mm | "
                                        "WhitePial=%.2f%% | PialLR=%.2f%% | alpha_wp=%.4f | alpha_lr=%.4f",
                                        epoch,
                                        score,
                                        rmse_mm,
                                        wp_pct,
                                        lr_pct,
                                        alpha_wp,
                                        alpha_lr,
                                    )
                            else:
                                log.info(
                                    "Epoch %d [ValScore] | Collision metrics not scheduled this epoch. "
                                    "RMSE-only score=%.4f.",
                                    epoch,
                                    score,
                                )

                            score_tensor.fill_(float(score))

                            if require_collision_for_best and not collision_available:
                                collision_error_tensor.fill_(1)
                                log.error(
                                    "Epoch %d [ValScore] | Complete collision metrics are required "
                                    "for model selection but were unavailable.",
                                    epoch,
                                )
                            else:
                                if tb_writer is not None:
                                    tb_writer.add_scalar("val/rmse_mm", rmse_mm, epoch)
                                    if collision_available:
                                        wp_total = lh_total + rh_total
                                        wp_hit = lh_hit + rh_hit
                                        wp_contacts = lh_contacts_sum + rh_contacts_sum
                                        tb_writer.add_scalar(
                                            "collisions/whitepial_pct_pairs_colliding_total",
                                            wp_pct,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/whitepial_num_pairs_colliding_total",
                                            wp_hit,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/whitepial_mean_contacts_all_total",
                                            wp_contacts / wp_total,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/whitepial_mean_contacts_hit_total",
                                            wp_contacts / max(wp_hit, 1.0),
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/piallr_pct_pairs_colliding",
                                            lr_pct,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/piallr_num_pairs_colliding",
                                            lr_hit,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/piallr_mean_contacts_all",
                                            lr_contacts_sum / lr_total,
                                            epoch,
                                        )
                                        tb_writer.add_scalar(
                                            "collisions/piallr_mean_contacts_hit",
                                            lr_contacts_sum / max(lr_hit, 1.0),
                                            epoch,
                                        )
                                        tb_writer.add_scalar("val/collision_aware_score", score, epoch)
                                        tb_writer.add_scalar("val/wp_collision_pct", wp_pct, epoch)
                                        tb_writer.add_scalar("val/pial_lr_collision_pct", lr_pct, epoch)
                                        tb_writer.add_scalar("val/best_collision_aware_score", best_score, epoch)

                                ckpt_rmse = os.path.join(
                                    out_root, "checkpoints", "deform_best_rmse.pth"
                                )
                                ckpt_rmse_full = os.path.join(
                                    out_root, "checkpoints", "deform_best_rmse_full.pth"
                                )
                                ckpt_model = os.path.join(
                                    out_root, "checkpoints", "deform_best_model.pth"
                                )
                                ckpt_model_full = os.path.join(
                                    out_root, "checkpoints", "deform_best_model_full.pth"
                                )

                                rmse_improved = rmse_mm < (best_rmse_seen - early_rmse_delta)
                                if rmse_improved:
                                    best_rmse_seen = rmse_mm
                                    best_rmse_epoch = epoch
                                    no_improve_rmse = 0
                                else:
                                    no_improve_rmse += 1

                                reasonable = rmse_mm <= best_rmse_seen * rmse_guardrail_rel
                                score_improved = False
                                score_save_message = None

                                if collision_available:
                                    if reasonable and score < (best_score - score_delta):
                                        best_score = score
                                        best_model_epoch = epoch
                                        no_improve = 0
                                        score_improved = True
                                        score_save_message = (
                                            "[BEST] Collision-aware model updated at epoch %d | "
                                            "Score=%.4f | RMSE=%.4f mm | WP=%.2f%% | PialLR=%.2f%% | "
                                            "Guardrail=%.4f mm | BestRMSE=%.4f mm -> %s",
                                            epoch,
                                            score,
                                            rmse_mm,
                                            wp_pct,
                                            lr_pct,
                                            best_rmse_seen * rmse_guardrail_rel,
                                            best_rmse_seen,
                                            ckpt_model,
                                        )
                                    else:
                                        no_improve += 1
                                        log.info(
                                            "Epoch %d [ValScore] | No score improvement | "
                                            "Score=%.4f | BestScore=%.4f | RMSE=%.4f mm | "
                                            "Reasonable=%s | BestModelEpoch=%d | no_improve=%d",
                                            epoch,
                                            score,
                                            best_score,
                                            rmse_mm,
                                            reasonable,
                                            best_model_epoch,
                                            no_improve,
                                        )
                                else:
                                    fallback_score = rmse_mm
                                    if fallback_score < (best_score - score_delta):
                                        best_score = fallback_score
                                        best_model_epoch = epoch
                                        no_improve = 0
                                        score_improved = True
                                        score_save_message = (
                                            "[BEST] RMSE-fallback model updated at epoch %d | "
                                            "RMSE=%.4f mm | collision_available=False | "
                                            "require_collision_for_best=False -> %s",
                                            epoch,
                                            rmse_mm,
                                            ckpt_model,
                                        )
                                    else:
                                        no_improve += 1
                                        log.info(
                                            "Epoch %d [ValScore] | No RMSE improvement "
                                            "(collision-unavailable fallback) | RMSE=%.4f mm | "
                                            "BestScore=%.4f | BestModelEpoch=%d | no_improve=%d",
                                            epoch,
                                            rmse_mm,
                                            best_score,
                                            best_model_epoch,
                                            no_improve,
                                        )

                                if rmse_improved:
                                    save_model_state(model, ckpt_rmse)
                                    save_full_checkpoint(
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        path=ckpt_rmse_full,
                                        epoch=epoch,
                                        best_score=best_score,
                                        best_rmse_seen=best_rmse_seen,
                                        best_model_epoch=best_model_epoch,
                                        best_rmse_epoch=best_rmse_epoch,
                                        no_improve=no_improve,
                                        no_improve_rmse=no_improve_rmse,
                                        cfg=cfg,
                                    )
                                    log.info(
                                        "[BEST] RMSE checkpoint updated at epoch %d | "
                                        "RMSE=%.4f mm -> %s",
                                        epoch,
                                        rmse_mm,
                                        ckpt_rmse,
                                    )

                                if score_improved:
                                    save_model_state(model, ckpt_model)
                                    save_full_checkpoint(
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        path=ckpt_model_full,
                                        epoch=epoch,
                                        best_score=best_score,
                                        best_rmse_seen=best_rmse_seen,
                                        best_model_epoch=best_model_epoch,
                                        best_rmse_epoch=best_rmse_epoch,
                                        no_improve=no_improve,
                                        no_improve_rmse=no_improve_rmse,
                                        cfg=cfg,
                                    )
                                    if score_save_message is not None:
                                        log.info(*score_save_message)

                                # Early stopping and LR scheduling monitor validation RMSE.
                                # Final model selection remains collision-aware when enabled.
                                if early_patience > 0 and no_improve_rmse >= early_patience:
                                    log.info(
                                        "[STOP] Early stopping after %d validation checks without "
                                        "RMSE improvement. BestScore=%.4f at epoch %d | "
                                        "BestRMSE=%.4f at epoch %d | no_improve(score)=%d "
                                        "no_improve(rmse)=%d",
                                        early_patience,
                                        best_score,
                                        best_model_epoch,
                                        best_rmse_seen,
                                        best_rmse_epoch,
                                        no_improve,
                                        no_improve_rmse,
                                    )
                                    stop_tensor.fill_(1)
                    except Exception as exc:
                        validation_error_tensor.fill_(1)
                        log.exception(
                            "Epoch %d [Val] | Rank-0 validation failed: %s",
                            epoch,
                            exc,
                        )
                if is_distributed:
                    dist.broadcast(rmse_tensor, src=0)
                    dist.broadcast(score_tensor, src=0)
                    dist.broadcast(collision_error_tensor, src=0)
                    dist.broadcast(validation_error_tensor, src=0)

                if validation_error_tensor.item() == 1:
                    raise RuntimeError(
                        "Validation coverage was incomplete. Every validation subject must "
                        "provide all four valid predicted and reference surfaces."
                    )
                if collision_error_tensor.item() == 1:
                    raise RuntimeError(
                        "Complete collision metrics were unavailable while "
                        "checkpoint.require_collision_for_best=True."
                    )

                # LR scheduler follows RMSE so noisy collision counts cannot trigger
                # a learning-rate drop while geometric accuracy is still improving.
                shared_rmse_mm = float(rmse_tensor.item())
                if math.isfinite(shared_rmse_mm):
                    scheduler.step(shared_rmse_mm)

                if rank == 0:
                    ckpt_last_full = os.path.join(
                        out_root, "checkpoints", "deform_last_full.pth"
                    )
                    save_full_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        path=ckpt_last_full,
                        epoch=epoch,
                        best_score=best_score,
                        best_rmse_seen=best_rmse_seen,
                        best_model_epoch=best_model_epoch,
                        best_rmse_epoch=best_rmse_epoch,
                        no_improve=no_improve,
                        no_improve_rmse=no_improve_rmse,
                        cfg=cfg,
                    )

                net.train()

            # Sync early-stop decision across ranks
            if is_distributed:
                dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() == 1:
                break

    finally:
        if tb_writer is not None:
            tb_writer.close()
        if file_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(file_handler)
            file_handler.flush()
            file_handler.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()

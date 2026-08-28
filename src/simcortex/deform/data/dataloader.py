from __future__ import annotations

import os
import logging
from typing import List, Sequence

import numpy as np
import nibabel as nib
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from simcortex.deform.utils.coords import (
    world_to_voxel,
    make_center_crop_pad_slices,
)

logger = logging.getLogger(__name__)


# ----------------------------
# BIDS-derivatives path helpers
# ----------------------------
def _ses(session_label: str) -> str:
    s = str(session_label).strip()
    if not s:
        raise ValueError("session_label must not be empty")
    return s if s.startswith("ses-") else f"ses-{s}"


def _sub(subject_label: str) -> str:
    s = str(subject_label).strip()
    if not s or s.lower() in {"nan", "none", "sub-"}:
        raise ValueError(f"Invalid subject label: {subject_label!r}")
    return s if s.startswith("sub-") else f"sub-{s}"


def _normalize_subjects(subjects: Sequence[str]) -> List[str]:
    normalized = [_sub(subject) for subject in subjects]
    duplicates = sorted({subject for subject in normalized if normalized.count(subject) > 1})
    if duplicates:
        raise ValueError(f"Duplicate subject labels are not allowed: {duplicates[:20]}")
    if not normalized:
        raise ValueError("subjects must contain at least one subject")
    return normalized


def _validate_surface_names(surface_names: Sequence[str]) -> List[str]:
    names = [str(name).strip() for name in surface_names]
    if not names:
        raise ValueError("surface_names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError(f"surface_names contains duplicates: {names}")
    unknown = sorted(set(names) - set(_SURF_MAP))
    if unknown:
        raise ValueError(f"Unknown surface names: {unknown}. Allowed: {sorted(_SURF_MAP)}")
    return names


def _validate_dataset_settings(
    inshape_dhw,
    prob_clip_min: float,
    prob_clip_max: float,
    prob_gamma: float,
) -> tuple[tuple[int, int, int], float, float, float]:
    inshape = tuple(int(value) for value in inshape_dhw)
    if len(inshape) != 3 or any(value <= 0 for value in inshape):
        raise ValueError(f"inshape_dhw must contain three positive integers, got {inshape}")

    clip_min = float(prob_clip_min)
    clip_max = float(prob_clip_max)
    gamma = float(prob_gamma)
    if not (0.0 <= clip_min <= clip_max <= 1.0):
        raise ValueError(
            "Probability clipping must satisfy 0 <= prob_clip_min <= "
            f"prob_clip_max <= 1, got {clip_min} and {clip_max}"
        )
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"prob_gamma must be finite and > 0, got {gamma}")
    return inshape, clip_min, clip_max, gamma


def _format_missing_inputs(kind: str, missing_by_subject) -> str:
    lines = [
        f"{kind} is missing required inputs for {len(missing_by_subject)} subject(s)."
    ]
    for subject, missing in missing_by_subject[:20]:
        lines.append(f"  {subject}:")
        lines.extend(f"    - {item}" for item in missing[:20])
    if len(missing_by_subject) > 20:
        lines.append(f"  ... and {len(missing_by_subject) - 20} more subject(s)")
    return "\n".join(lines)

def mni_t1_path(preproc_root: str, subj: str, session_label: str, space: str) -> str:
    ses = _ses(session_label)
    return os.path.join(
        preproc_root, subj, ses, "anat",
        f"{subj}_{ses}_space-{space}_desc-preproc_T1w.nii.gz",
    )


def ribbon_prob_path(initsurf_root: str, subj: str, session_label: str, space: str) -> str:
    ses = _ses(session_label)
    return os.path.join(
        initsurf_root, subj, ses, "anat",
        f"{subj}_{ses}_space-{space}_desc-ribbon_prob.nii.gz",
    )


_SURF_MAP = {
    "lh_pial":  ("L", "pial"),
    "lh_white": ("L", "white"),
    "rh_pial":  ("R", "pial"),
    "rh_white": ("R", "white"),
}


def surf_path(root: str, subj: str, session_label: str, space: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, surf = _SURF_MAP[surf_name]
    return os.path.join(
        root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_hemi-{hemi}_{surf}.surf.ply",
    )


_FIXED_TEMPLATE_FILENAMES = {
    "lh_pial": "lh_pial_smoothed.ply",
    "lh_white": "lh_white_smoothed.ply",
    "rh_pial": "rh_pial_smoothed.ply",
    "rh_white": "rh_white_smoothed.ply",
}


def fixed_surf_path(fixed_template_root: str, surf_name: str) -> str:
    if surf_name not in _FIXED_TEMPLATE_FILENAMES:
        raise KeyError(
            f"Unknown fixed template surface name: {surf_name}. "
            f"Allowed: {sorted(_FIXED_TEMPLATE_FILENAMES)}"
        )
    return os.path.join(
        str(fixed_template_root),
        _FIXED_TEMPLATE_FILENAMES[surf_name],
    )


# ----------------------------
# IO helpers
# ----------------------------
def read_nii(path: str):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    aff = np.asarray(nii.affine, dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}, got shape={vol.shape}")
    if aff.shape != (4, 4) or not np.isfinite(aff).all():
        raise ValueError(f"Invalid affine in {path}: shape={aff.shape}")
    det = float(np.linalg.det(aff[:3, :3]))
    if not np.isfinite(det) or abs(det) < 1e-8:
        raise ValueError(f"Singular or invalid affine in {path}: det={det}")
    return vol, aff


def _validate_mesh_arrays(v: np.ndarray, f: np.ndarray, path: str) -> None:
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Invalid vertices in {path}: shape={v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Invalid faces in {path}: shape={f.shape}")
    if v.shape[0] == 0 or f.shape[0] == 0:
        raise ValueError(f"Empty mesh in {path}: V={v.shape[0]}, F={f.shape[0]}")
    if not np.isfinite(v).all():
        raise ValueError(f"Non-finite vertices in {path}")
    if f.min() < 0 or f.max() >= v.shape[0]:
        raise ValueError(
            f"Invalid face indices in {path}: min={f.min()}, max={f.max()}, V={v.shape[0]}"
        )


def read_mesh(path: str):
    m = trimesh.load(path, process=False)

    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values()]
        if len(geoms) == 0:
            raise ValueError(f"Empty trimesh.Scene (no geometry) loaded from: {path}")
        m = trimesh.util.concatenate(geoms)

    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.faces, dtype=np.int64)
    _validate_mesh_arrays(v, f, path)
    return v, f

def normalize_mri_mean_std(mri: np.ndarray) -> np.ndarray:
    if not np.isfinite(mri).all():
        raise ValueError("MRI volume contains non-finite values")

    mask = mri != 0
    if mask.sum() < 100:
        mean = float(mri.mean())
        std = float(mri.std())
    else:
        mean = float(mri[mask].mean())
        std = float(mri[mask].std())

    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError(f"MRI normalization statistics are invalid: mean={mean}, std={std}")
    std = max(std, 1e-6)
    normalized = ((mri - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("MRI normalization produced non-finite values")
    return normalized



class CSRDeformDataset(Dataset):
    """
    Returns per subject:
      vol: (C,D,H,W) float32  [MRI, RIBBON_PROB]
      affine: (4,4) float32 (vox->world)
      shift_ijk: (3,) float32
      init_verts_vox[surf], init_faces[surf]
      gt_verts_vox[surf], gt_faces[surf]
    """

    def __init__(
        self,
        preproc_root: str,
        initsurf_root: str | None,
        subjects: List[str],
        session_label: str,
        space: str,
        surface_names,
        inshape_dhw,
        prob_clip_min: float = 0.0,
        prob_clip_max: float = 1.0,
        prob_gamma: float = 1.0,
        aug: bool = False,  # backward-compatible no-op; augmentation is handled in train.py
        strict_missing: bool = True,
        use_probability_map: bool = True,
        use_fixed_initial_surface: bool = False,
        fixed_template_root: str | None = None,
    ):
        self.preproc_root = str(preproc_root)
        self.initsurf_root = (
            None
            if initsurf_root in (None, "")
            else str(initsurf_root)
        )
        self.subjects = _normalize_subjects(subjects)
        self.session_label = str(session_label).strip()
        self.space = str(space).strip()
        if not self.space:
            raise ValueError("space must not be empty")

        self.surface_names = _validate_surface_names(surface_names)
        (
            self.inshape,
            self.prob_clip_min,
            self.prob_clip_max,
            self.prob_gamma,
        ) = _validate_dataset_settings(
            inshape_dhw,
            prob_clip_min,
            prob_clip_max,
            prob_gamma,
        )

        self.strict_missing = bool(strict_missing)
        self.use_probability_map = bool(use_probability_map)
        self.use_fixed_initial_surface = bool(use_fixed_initial_surface)
        self.fixed_template_root = (
            None
            if fixed_template_root in (None, "")
            else str(fixed_template_root)
        )

        if self.use_fixed_initial_surface and self.fixed_template_root is None:
            raise ValueError(
                "use_fixed_initial_surface=True requires fixed_template_root."
            )

        needs_subject_initsurf = (
            self.use_probability_map
            or not self.use_fixed_initial_surface
        )
        if needs_subject_initsurf and self.initsurf_root is None:
            raise ValueError(
                "initsurf_root is required when using a subject-specific "
                "probability map or subject-specific initial surfaces."
            )

        self.samples = []
        missing_by_subject = []

        for subj in self.subjects:
            mri_path = mni_t1_path(
                self.preproc_root,
                subj,
                self.session_label,
                self.space,
            )

            prob_path = None
            if self.use_probability_map:
                prob_path = ribbon_prob_path(
                    self.initsurf_root,
                    subj,
                    self.session_label,
                    self.space,
                )

            gt_paths = {
                s: surf_path(
                    self.preproc_root,
                    subj,
                    self.session_label,
                    self.space,
                    s,
                )
                for s in self.surface_names
            }

            if self.use_fixed_initial_surface:
                ini_paths = {
                    s: fixed_surf_path(self.fixed_template_root, s)
                    for s in self.surface_names
                }
            else:
                ini_paths = {
                    s: surf_path(
                        self.initsurf_root,
                        subj,
                        self.session_label,
                        self.space,
                        s,
                    )
                    for s in self.surface_names
                }

            missing = []

            if not os.path.isfile(mri_path):
                missing.append(mri_path)

            if prob_path is not None and not os.path.isfile(prob_path):
                missing.append(prob_path)

            for s in self.surface_names:
                if not os.path.isfile(gt_paths[s]):
                    missing.append(gt_paths[s])
                if not os.path.isfile(ini_paths[s]):
                    missing.append(ini_paths[s])

            if missing:
                missing_by_subject.append((subj, missing))
                continue

            self.samples.append(
                (subj, mri_path, prob_path, gt_paths, ini_paths)
            )

        if missing_by_subject:
            message = _format_missing_inputs(
                "CSRDeformDataset",
                missing_by_subject,
            )
            if self.strict_missing:
                raise FileNotFoundError(message)
            logger.warning("%s", message)

        if len(self.samples) == 0:
            raise RuntimeError(
                "CSRDeformDataset found zero valid subjects. "
                "Check preproc_root, subject IDs, session_label, space, "
                "fixed-template settings, initsurf_root when required, "
                "and surface file names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj, mri_path, prob_path, gt_paths, ini_paths = self.samples[idx]

        mri, affine = read_nii(mri_path)
        prob = None

        if self.use_probability_map:
            prob, prob_affine = read_nii(prob_path)

            if not np.allclose(prob_affine, affine, atol=1e-4, rtol=0.0):
                raise ValueError(
                    f"PROB/MRI affine mismatch for {subj}: "
                    f"prob_affine={prob_affine}, mri_affine={affine}"
                )

            if prob.shape != mri.shape:
                raise ValueError(
                    f"PROB/MRI shape mismatch for {subj}: "
                    f"prob={prob.shape}, mri={mri.shape}"
                )

        mri = normalize_mri_mean_std(mri)

        if self.use_probability_map:
            prob = np.nan_to_num(
                prob,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).astype(np.float32)

            if self.prob_clip_min > 0:
                prob[prob < self.prob_clip_min] = 0.0

            prob = np.clip(
                prob,
                0.0,
                self.prob_clip_max,
            ).astype(np.float32)

            if abs(self.prob_gamma - 1.0) > 1e-6:
                prob = np.power(
                    prob,
                    self.prob_gamma,
                ).astype(np.float32)

        D0, H0, W0 = mri.shape
        D1, H1, W1 = self.inshape

        crop_slices, pad_before, pad_after, crop_before = (
            make_center_crop_pad_slices(
                (D0, H0, W0),
                (D1, H1, W1),
            )
        )

        mri_c = mri[
            crop_slices[0],
            crop_slices[1],
            crop_slices[2],
        ]

        prob_c = (
            prob[
                crop_slices[0],
                crop_slices[1],
                crop_slices[2],
            ]
            if self.use_probability_map
            else None
        )

        pbD, pbH, pbW = pad_before
        paD, paH, paW = pad_after

        mri_t = torch.from_numpy(mri_c)[None, None]
        prob_t = (
            torch.from_numpy(prob_c)[None, None]
            if self.use_probability_map
            else None
        )

        mri_t = F.pad(
            mri_t,
            (pbW, paW, pbH, paH, pbD, paD),
            mode="replicate",
        )

        if self.use_probability_map:
            prob_t = F.pad(
                prob_t,
                (pbW, paW, pbH, paH, pbD, paD),
                mode="constant",
                value=0.0,
            )

        mri_out = mri_t[0, 0].numpy()
        prob_out = (
            prob_t[0, 0].numpy()
            if self.use_probability_map
            else None
        )

        if mri_out.shape != self.inshape:
            raise ValueError(
                f"Internal crop/pad error for subject '{subj}': "
                f"got {mri_out.shape}, expected {self.inshape}"
            )

        shift_ijk = (
            np.array(pad_before, dtype=np.float32)
            - np.array(crop_before, dtype=np.float32)
        )

        A = torch.from_numpy(affine).float()

        init_verts_vox = {}
        init_faces = {}
        gt_verts_vox = {}
        gt_faces = {}

        for s in self.surface_names:
            v_ini_mm, f_ini = read_mesh(ini_paths[s])
            v_gt_mm, f_gt = read_mesh(gt_paths[s])

            v_ini = world_to_voxel(
                torch.from_numpy(v_ini_mm).float(),
                A,
            ).numpy()

            v_gt = world_to_voxel(
                torch.from_numpy(v_gt_mm).float(),
                A,
            ).numpy()

            v_ini = (v_ini + shift_ijk).astype(np.float32)
            v_gt = (v_gt + shift_ijk).astype(np.float32)

            init_verts_vox[s] = torch.from_numpy(v_ini).float()
            init_faces[s] = torch.from_numpy(f_ini).long()
            gt_verts_vox[s] = torch.from_numpy(v_gt).float()
            gt_faces[s] = torch.from_numpy(f_gt).long()

        channels = [
            torch.from_numpy(mri_out).float()
        ]

        if self.use_probability_map:
            channels.append(
                torch.from_numpy(prob_out).float()
            )

        vol = torch.stack(channels, dim=0)

        return {
            "subject": subj,
            "vol": vol,
            "affine": torch.from_numpy(affine).float(),
            "shift_ijk": torch.from_numpy(shift_ijk).float(),
            "init_verts_vox": init_verts_vox,
            "init_faces": init_faces,
            "gt_verts_vox": gt_verts_vox,
            "gt_faces": gt_faces,
        }




class CSRDeformInferDataset(Dataset):
    """
    Inference-only dataset for SurfDeform.

    Required inputs per subject:
      - MNI-space preprocessed T1w image from sc-preproc
      - optional ribbon probability map from sc-initsurf
      - subject-specific or shared fixed initial surfaces
    """

    def __init__(
        self,
        preproc_root: str,
        initsurf_root: str | None,
        subjects: List[str],
        session_label: str,
        space: str,
        surface_names,
        inshape_dhw,
        prob_clip_min: float = 0.0,
        prob_clip_max: float = 1.0,
        prob_gamma: float = 1.0,
        strict_missing: bool = True,
        use_probability_map: bool = True,
        use_fixed_initial_surface: bool = False,
        fixed_template_root: str | None = None,
    ):
        self.preproc_root = str(preproc_root)
        self.initsurf_root = (
            None
            if initsurf_root in (None, "")
            else str(initsurf_root)
        )
        self.subjects = _normalize_subjects(subjects)
        self.session_label = str(session_label).strip()
        self.space = str(space).strip()
        if not self.space:
            raise ValueError("space must not be empty")

        self.surface_names = _validate_surface_names(surface_names)
        (
            self.inshape,
            self.prob_clip_min,
            self.prob_clip_max,
            self.prob_gamma,
        ) = _validate_dataset_settings(
            inshape_dhw,
            prob_clip_min,
            prob_clip_max,
            prob_gamma,
        )

        self.strict_missing = bool(strict_missing)
        self.use_probability_map = bool(use_probability_map)
        self.use_fixed_initial_surface = bool(use_fixed_initial_surface)
        self.fixed_template_root = (
            None
            if fixed_template_root in (None, "")
            else str(fixed_template_root)
        )

        if self.use_fixed_initial_surface and self.fixed_template_root is None:
            raise ValueError(
                "use_fixed_initial_surface=True requires fixed_template_root."
            )

        needs_subject_initsurf = (
            self.use_probability_map
            or not self.use_fixed_initial_surface
        )
        if needs_subject_initsurf and self.initsurf_root is None:
            raise ValueError(
                "initsurf_root is required when using a subject-specific "
                "probability map or subject-specific initial surfaces."
            )

        self.samples = []
        missing_by_subject = []

        for subj in self.subjects:
            mri_path = mni_t1_path(
                self.preproc_root,
                subj,
                self.session_label,
                self.space,
            )

            prob_path = None
            if self.use_probability_map:
                prob_path = ribbon_prob_path(
                    self.initsurf_root,
                    subj,
                    self.session_label,
                    self.space,
                )

            if self.use_fixed_initial_surface:
                ini_paths = {
                    s: fixed_surf_path(self.fixed_template_root, s)
                    for s in self.surface_names
                }
            else:
                ini_paths = {
                    s: surf_path(
                        self.initsurf_root,
                        subj,
                        self.session_label,
                        self.space,
                        s,
                    )
                    for s in self.surface_names
                }

            missing = []

            if not os.path.isfile(mri_path):
                missing.append(mri_path)

            if prob_path is not None and not os.path.isfile(prob_path):
                missing.append(prob_path)

            for s in self.surface_names:
                if not os.path.isfile(ini_paths[s]):
                    missing.append(ini_paths[s])

            if missing:
                missing_by_subject.append((subj, missing))
                continue

            self.samples.append(
                (subj, mri_path, prob_path, ini_paths)
            )

        if missing_by_subject:
            message = _format_missing_inputs(
                "CSRDeformInferDataset",
                missing_by_subject,
            )
            if self.strict_missing:
                raise FileNotFoundError(message)
            logger.warning("%s", message)

        if len(self.samples) == 0:
            raise RuntimeError(
                "CSRDeformInferDataset found zero valid subjects. "
                "Check preproc_root, subject IDs, session_label, space, "
                "fixed-template settings, and initsurf_root when required."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj, mri_path, prob_path, ini_paths = self.samples[idx]

        mri, affine = read_nii(mri_path)
        prob = None

        if self.use_probability_map:
            prob, prob_affine = read_nii(prob_path)

            if not np.allclose(prob_affine, affine, atol=1e-4, rtol=0.0):
                raise ValueError(
                    f"PROB/MRI affine mismatch for {subj}: "
                    f"prob_affine={prob_affine}, mri_affine={affine}"
                )

            if prob.shape != mri.shape:
                raise ValueError(
                    f"PROB/MRI shape mismatch for {subj}: "
                    f"prob={prob.shape}, mri={mri.shape}"
                )

        mri = normalize_mri_mean_std(mri)

        if self.use_probability_map:
            prob = np.nan_to_num(
                prob,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).astype(np.float32)

            if self.prob_clip_min > 0:
                prob[prob < self.prob_clip_min] = 0.0

            prob = np.clip(
                prob,
                0.0,
                self.prob_clip_max,
            ).astype(np.float32)

            if abs(self.prob_gamma - 1.0) > 1e-6:
                prob = np.power(
                    prob,
                    self.prob_gamma,
                ).astype(np.float32)

        D0, H0, W0 = mri.shape
        D1, H1, W1 = self.inshape

        crop_slices, pad_before, pad_after, crop_before = (
            make_center_crop_pad_slices(
                (D0, H0, W0),
                (D1, H1, W1),
            )
        )

        mri_c = mri[
            crop_slices[0],
            crop_slices[1],
            crop_slices[2],
        ]

        prob_c = (
            prob[
                crop_slices[0],
                crop_slices[1],
                crop_slices[2],
            ]
            if self.use_probability_map
            else None
        )

        pbD, pbH, pbW = pad_before
        paD, paH, paW = pad_after

        mri_t = torch.from_numpy(mri_c)[None, None]
        prob_t = (
            torch.from_numpy(prob_c)[None, None]
            if self.use_probability_map
            else None
        )

        mri_t = F.pad(
            mri_t,
            (pbW, paW, pbH, paH, pbD, paD),
            mode="replicate",
        )

        if self.use_probability_map:
            prob_t = F.pad(
                prob_t,
                (pbW, paW, pbH, paH, pbD, paD),
                mode="constant",
                value=0.0,
            )

        mri_out = mri_t[0, 0].numpy()
        prob_out = (
            prob_t[0, 0].numpy()
            if self.use_probability_map
            else None
        )

        if mri_out.shape != self.inshape:
            raise ValueError(
                f"Internal crop/pad error for subject '{subj}': "
                f"got {mri_out.shape}, expected {self.inshape}"
            )

        shift_ijk = (
            np.array(pad_before, dtype=np.float32)
            - np.array(crop_before, dtype=np.float32)
        )

        A = torch.from_numpy(affine).float()

        init_verts_vox = {}
        init_faces = {}

        for s in self.surface_names:
            v_ini_mm, f_ini = read_mesh(ini_paths[s])

            v_ini = world_to_voxel(
                torch.from_numpy(v_ini_mm).float(),
                A,
            ).numpy()

            v_ini = (v_ini + shift_ijk).astype(np.float32)

            init_verts_vox[s] = torch.from_numpy(v_ini).float()
            init_faces[s] = torch.from_numpy(f_ini).long()

        channels = [
            torch.from_numpy(mri_out).float()
        ]

        if self.use_probability_map:
            channels.append(
                torch.from_numpy(prob_out).float()
            )

        vol = torch.stack(channels, dim=0)

        return {
            "subject": subj,
            "vol": vol,
            "affine": torch.from_numpy(affine).float(),
            "shift_ijk": torch.from_numpy(shift_ijk).float(),
            "init_verts_vox": init_verts_vox,
            "init_faces": init_faces,
        }


def collate_csr_deform_infer(batch_list):
    return {
        "subject": [b["subject"] for b in batch_list],
        "vol": torch.stack([b["vol"] for b in batch_list], dim=0),
        "affine": torch.stack([b["affine"] for b in batch_list], dim=0),
        "shift_ijk": torch.stack([b["shift_ijk"] for b in batch_list], dim=0),
        "init_verts_vox": [b["init_verts_vox"] for b in batch_list],
        "init_faces": [b["init_faces"] for b in batch_list],
    }


def collate_csr_deform(batch_list):
    return {
        "subject": [b["subject"] for b in batch_list],
        "vol": torch.stack([b["vol"] for b in batch_list], dim=0),
        "affine": torch.stack([b["affine"] for b in batch_list], dim=0),
        "shift_ijk": torch.stack([b["shift_ijk"] for b in batch_list], dim=0),
        "init_verts_vox": [b["init_verts_vox"] for b in batch_list],
        "init_faces": [b["init_faces"] for b in batch_list],
        "gt_verts_vox": [b["gt_verts_vox"] for b in batch_list],
        "gt_faces": [b["gt_faces"] for b in batch_list],
    }

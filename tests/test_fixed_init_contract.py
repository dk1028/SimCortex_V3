from pathlib import Path
import inspect

from omegaconf import OmegaConf

from simcortex.cli.main import app
from simcortex.deform.models.surfdeform import SurfDeform


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = REPO_ROOT / "src/simcortex/configs/deform/train.yaml"
BASE_INFER = REPO_ROOT / "src/simcortex/configs/deform/inference.yaml"
FIXED_CONFIG_DIR = REPO_ROOT / "configs/fixed_init"


def test_base_train_is_mri_only_fixed_initialization() -> None:
    cfg = OmegaConf.load(BASE_TRAIN)
    assert bool(cfg.dataset.use_probability_map) is False
    assert bool(cfg.dataset.use_fixed_initial_surface) is True
    assert int(cfg.model.c_in) == 1
    assert int(cfg.model.n_steps) == 8


def test_base_inference_is_mri_only_fixed_initialization() -> None:
    cfg = OmegaConf.load(BASE_INFER)
    assert bool(cfg.dataset.use_probability_map) is False
    assert bool(cfg.dataset.use_fixed_initial_surface) is True
    assert int(cfg.model.c_in) == 1
    assert bool(cfg.model.strict_load) is True


def test_all_five_final_overlays_use_fixed_initialization(monkeypatch) -> None:
    monkeypatch.setenv(
        "SIMCORTEX_RESOURCES_ROOT",
        "/tmp/simcortex-fixedinit-resources",
    )
    monkeypatch.setenv(
        "SIMCORTEX_RUNS_ROOT",
        "/tmp/simcortex-fixedinit-runs",
    )

    names = ["sphere", "random", "original150k", "curv0", "curv1"]

    for name in names:
        cfg = OmegaConf.load(FIXED_CONFIG_DIR / f"{name}.yaml")

        assert bool(cfg.dataset.use_probability_map) is False
        assert bool(cfg.dataset.use_fixed_initial_surface) is True
        assert str(cfg.dataset.fixed_template_root).startswith(
            "/tmp/simcortex-fixedinit-resources/"
        )
        assert int(cfg.model.c_in) == 1
        assert int(cfg.trainer.seed) == 2025


def test_model_signature_is_single_encoder_contract() -> None:
    params = inspect.signature(SurfDeform).parameters
    assert params["C_in"].default == 1
    assert "geom_ratio" not in params
    assert "geom_depth" not in params
    assert "gate_init" not in params


def test_cli_exists_for_deformation_only() -> None:
    assert app is not None

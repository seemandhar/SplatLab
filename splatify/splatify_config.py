"""
Configuration for the Splatify Claude Agents pipeline.

All paths reference the gs_template project which contains the modular
Gaussian Splatting framework with 20+ paper implementations, shared
rendering helpers, loss functions, and DataManagers.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_auth_logged = False


def get_claude_auth_env() -> dict[str, str]:
    """Build env dict for Claude Agent SDK, preferring OAuth over API key."""
    global _auth_logged
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if auth_token:
        if not _auth_logged:
            logger.info("Using Anthropic OAuth authentication")
            _auth_logged = True
        return {"ANTHROPIC_AUTH_TOKEN": auth_token}
    elif api_key:
        if not _auth_logged:
            logger.info("OAuth not configured, falling back to API key")
            _auth_logged = True
        return {"ANTHROPIC_API_KEY": api_key}
    else:
        if not _auth_logged:
            logger.warning(
                "No Anthropic auth configured. Set ANTHROPIC_AUTH_TOKEN (OAuth) or ANTHROPIC_API_KEY."
            )
            _auth_logged = True
        return {}


# Dataset-aware PSNR baselines (approximate splatfacto baselines per dataset)
# These are used as reference thresholds during PSNR review — a method should
# get within ~2 dB of the baseline to be considered "working correctly".
DATASET_PSNR_BASELINES: dict[str, dict[str, float]] = {
    "mipnerf360": {
        "bicycle": 25.0, "bonsai": 31.5, "counter": 28.5, "garden": 27.0,
        "kitchen": 30.5, "room": 30.5, "stump": 26.5, "flowers": 21.5,
        "treehill": 22.5, "_default": 27.0,
    },
    "llff": {
        "fern": 25.0, "flower": 27.5, "fortress": 31.0, "horns": 27.5,
        "leaves": 21.0, "orchids": 20.0, "room": 32.0, "trex": 26.5,
        "_default": 25.0,
    },
}

DATASET_DATA_ROOTS: dict[str, str] = {
    "mipnerf360": "data/mipnerf360",
    "llff": "data/nerf_llff_data",
}


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Source project: the gs_template framework with 20+ implemented methods
    gs_template_root: Path = field(
        default_factory=lambda: Path("splatify/gs_template")
    )

    # Dataset / scene configuration
    dataset: str = "mipnerf360"        # "mipnerf360" or "llff"
    scenes: list[str] = field(default_factory=lambda: ["garden"])

    # Paths derived from gs_template
    @property
    def model_template(self) -> Path:
        """Base model template (GSTemplateModel)."""
        return self.gs_template_root / "model.py"

    @property
    def rendering_helpers(self) -> Path:
        """Shared rendering helpers (prepare_camera, rasterize_gaussians, etc.)."""
        return self.gs_template_root / "rendering.py"

    @property
    def loss_library(self) -> Path:
        """Shared loss function library."""
        return self.gs_template_root / "losses.py"

    @property
    def datamanager_template(self) -> Path:
        """Custom DataManager templates (DepthDataManager, etc.)."""
        return self.gs_template_root / "datamanagers.py"

    @property
    def shading_utils(self) -> Path:
        """Shading utilities (SH, normals, cubemaps)."""
        return self.gs_template_root / "shading.py"

    @property
    def optimizer_template(self) -> Path:
        """Custom optimizer implementations (BOGausSAdam)."""
        return self.gs_template_root / "optimizers.py"

    @property
    def configs_registry(self) -> Path:
        """MethodSpecification registry showing how methods are registered."""
        return self.gs_template_root / "configs.py"

    @property
    def init_template(self) -> Path:
        """Package __init__.py showing exports."""
        return self.gs_template_root / "__init__.py"

    @property
    def methods_dir(self) -> Path:
        """Directory of 20+ implemented paper methods (in-context examples)."""
        return self.gs_template_root / "methods"

    @property
    def cfg_md(self) -> Path:
        """CFG.md — Context-Free Grammar for GS paper-to-code mapping."""
        return self.gs_template_root.parent / "CFG.md"

    @property
    def eval_script(self) -> Path:
        """eval.py script for full training + evaluation."""
        return self.gs_template_root.parent / "eval.py"

    @property
    def read_tb_script(self) -> Path:
        """read_tb.py script for extracting PSNR from TensorBoard logs."""
        return self.gs_template_root.parent / "read_tb.py"

    @property
    def default_dataset(self) -> str:
        """Default dataset for smoke testing (garden scene from Mip-NeRF 360)."""
        return str(Path("data/mipnerf360/garden"))

    # Output directories (in this project)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    workspace_dir: Path = field(default_factory=lambda: Path(__file__).parent / "workspace")
    generated_dir: Path = field(default_factory=lambda: Path(__file__).parent / "generated")

    # Pipeline options
    enable_review: bool = True
    enable_validation: bool = True
    enable_smoke_test: bool = True
    enable_training: bool = False
    incremental_generation: bool = True
    max_review_iterations: int = 2
    max_debug_iterations: int = 3
    max_training_review_iterations: int = 2
    expected_psnr: float | None = None  # Expected PSNR for custom dataset (overrides default 25 dB)

    # Nerfstudio / gsplat
    conda_env: str = "nerfstudio"
    default_max_iters: int = 30000
    num_views: int = -1  # Number of training views (-1 = all, or 3/6/9/12/24)
    viewer_port: int | None = None  # --viewer.websocket-port for ns-train (None = auto)

    # Required output files for a new method
    # A method implementation consists of:
    #   1. methods/<method_name>.py  — Model + Config class
    #   2. Updated configs.py entry  — MethodSpecification registration
    #   3. Updated methods/__init__.py — Export new classes
    #   4. Updated __init__.py — Export to package level
    #   5. README.md — Method documentation
    # But for the pipeline, we generate a STANDALONE installable package:
    # The package directory is named after the method (e.g. "dashgaussian/")
    # Use get_required_files(method_pkg) to get the actual file list.
    REQUIRED_FILES: list = field(default_factory=lambda: [
        "{method_pkg}/__init__.py",
        "{method_pkg}/model.py",
        "{method_pkg}/losses.py",
        "{method_pkg}/rendering.py",
        "{method_pkg}/configs.py",
        "{method_pkg}/datamanager.py",
        "README.md",
        "pyproject.toml",
    ])

    def get_required_files(self, method_pkg: str) -> list[str]:
        """Return required files with the actual method package name substituted."""
        return [f.format(method_pkg=method_pkg) for f in self.REQUIRED_FILES]

    def get_psnr_target(self) -> float:
        """Get the PSNR target for the current dataset/scenes config.

        Uses explicit expected_psnr if set, otherwise looks up the baseline
        for the first scene in the dataset and subtracts 2 dB margin.
        """
        if self.expected_psnr is not None:
            return self.expected_psnr
        baselines = DATASET_PSNR_BASELINES.get(self.dataset, {})
        scene = self.scenes[0] if self.scenes else "_default"
        baseline = baselines.get(scene, baselines.get("_default", 25.0))
        return round(baseline - 2.0, 1)

    def get_data_root(self) -> str:
        """Get the data root directory for the current dataset."""
        return DATASET_DATA_ROOTS.get(self.dataset,
            "data/mipnerf360")

    def get_scenes_str(self) -> str:
        """Get scenes as a space-separated string for eval.py CLI."""
        return " ".join(self.scenes) if self.scenes else "garden"

    def ensure_dirs(self):
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

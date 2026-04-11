import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Optional[Path] = None
    train_index_path: Optional[Path] = None
    val_index_path: Optional[Path] = None
    test_index_path: Optional[Path] = None
    num_context_views: int = 1


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        index_path = self._resolve_index_path(stage)
        if index_path is None:
            raise RuntimeError(
                f"Evaluation sampler requires an index path for stage={stage}. "
                f"Got train={cfg.train_index_path}, val={cfg.val_index_path}, test={cfg.test_index_path}, default={cfg.index_path}."
            )

        dacite_config = Config(cast=[tuple])
        with index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }

    def _resolve_index_path(self, stage: Stage) -> Optional[Path]:
        if stage == "train" and self.cfg.train_index_path is not None:
            return self.cfg.train_index_path
        if stage == "val" and self.cfg.val_index_path is not None:
            return self.cfg.val_index_path
        if stage == "test" and self.cfg.test_index_path is not None:
            return self.cfg.test_index_path
        return self.cfg.index_path

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0

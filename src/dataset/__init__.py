from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_tum_orb import DatasetTUMORB, DatasetTUMORBCfg
from .dataset_tartanair import DatasetTartanAir, DatasetTartanAirCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "tum_orb": DatasetTUMORB,
    "tartanair": DatasetTartanAir,
}

DatasetCfg = DatasetRE10kCfg | DatasetTUMORBCfg | DatasetTartanAirCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    print(f">>> get_dataset called: stage={stage}, dataset={cfg.name}")
    return DATASETS[cfg.name](cfg, stage, view_sampler)

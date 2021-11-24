from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
}

from .dataset import SceneFlowDataset, KITTIDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset
}

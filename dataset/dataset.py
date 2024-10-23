from dataset.team_features import TeamFeatures
from torch.utils.data import Dataset
import torch


class TeamDataset(Dataset):
    def __init__(
        self, features: TeamFeatures, device: torch.device = torch.device("cpu")
    ):
        super(TeamDataset, self).__init__()
        self.data = {"features": features.features}

    def __len__(self):
        return len(self.data["features"])

    def __getitem__(self, idx: int):
        return self.data["features"][idx]

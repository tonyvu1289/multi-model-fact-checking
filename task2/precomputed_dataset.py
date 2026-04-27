import torch


class PrecomputedClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, precomputed_path):
        # precomputed_path: path to a .pt file containing a list of dicts
        self._data = torch.load(precomputed_path)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def to_list(self):
        return self._data

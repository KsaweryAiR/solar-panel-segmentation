import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class PanelsDataset(Dataset):
    def __init__(self, images_paths: list[Path], transforms):
        self._images_paths = images_paths
        self._transforms = transforms

    def __len__(self):
        return len(self._images_paths)

    def __getitem__(self, index: int):
        image_path = self._images_paths[index]
        image = np.asarray(Image.open(image_path).convert('RGB'))

        mask_path = image_path.parent.parent / 'labels' / image_path.name
        mask = np.asarray(Image.open(mask_path))

        transformed = self._transforms(image=image, mask=mask)

        return transformed['image'], transformed['mask'].type(torch.float32).unsqueeze(0)

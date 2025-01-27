from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import albumentations.pytorch.transforms
from segmentation_package.datasets.panels import PanelsDataset
import lightning.pytorch as pl

class PanelsDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augmentations = A.Compose([
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.transforms.ToTensorV2(),
        ])
        self.transforms = A.Compose([
            A.Resize(width=256, height=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.transforms.ToTensorV2(),
        ])

    def setup(self, stage=None):
        dataset_path = Path("C:/Users/Antonina/PycharmProjects/zpo_panele_psy_pyth12/dane2")
        #dataset_path = Path("D:/panele_baza/ign/dane3")
        #dataset_path = Path("C:/Users/Antonina/PycharmProjects/zpo_panele_psy/dane5")
        train_paths = sorted((dataset_path / 'train' / 'images').glob('*.png'))
        test_paths = sorted((dataset_path / 'test' / 'images').glob('*.png'))

        train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=42)

        self.train_dataset = PanelsDataset(train_paths, self.augmentations)
        self.val_dataset = PanelsDataset(val_paths, self.transforms)
        self.test_dataset = PanelsDataset(test_paths, self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

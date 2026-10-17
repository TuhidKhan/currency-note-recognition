import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Standard normalization values for pretrained ImageNet models (like ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CurrencyDataset(Dataset):
    """
    Robust PyTorch Dataset for Currency Note Recognition.
    Automatically discovers all images in:
      dataset / <Currency> / <Denomination> / <Front or Back> / *.jpg
    """
    def __init__(self, root_dir="dataset", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []       # List of (image_path, class_idx)
        self.classes = []       # List of unique class names (e.g. 'PKR_500')
        self.class_to_idx = {}  # Dictionary mapping class name to integer index

        self._load_dataset()

    def _load_dataset(self):
        # 1. Discover all unique classes (e.g. Afghani_10, PKR_500)
        found_classes = set()
        for currency in sorted(os.listdir(self.root_dir)):
            currency_path = os.path.join(self.root_dir, currency)
            if not os.path.isdir(currency_path):
                continue

            for denom in sorted(os.listdir(currency_path), key=lambda x: int(x) if x.isdigit() else x):
                denom_path = os.path.join(currency_path, denom)
                if not os.path.isdir(denom_path):
                    continue

                class_name = f"{currency}_{denom}"
                found_classes.add(class_name)

        # Sort classes logically so mapping is always consistent
        self.classes = sorted(list(found_classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 2. Collect all image file paths and assign their label index
        for class_name, class_idx in self.class_to_idx.items():
            currency, denom = class_name.split("_")
            denom_dir = os.path.join(self.root_dir, currency, denom)

            # Traverse Front/Back folders or directly inside denom_dir
            for root, _, files in os.walk(denom_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(root, file)
                        self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Optimization: Pre-downscale scans exceeding 800px for 10x faster augmentations on CPU
        if max(image.size) > 800:
            image.thumbnail((800, 800), Image.Resampling.BILINEAR)

        if self.transform:
            image = self.transform(image)

        return image, class_idx


def get_transforms():
    """
    Returns data pipelines:
    - Train: Multi-scale crop, rotation, flip, and color jitter to learn folded notes and real-world lighting
    - Val: Standard resizing and normalization
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(224, 224),
            scale=(0.35, 1.0),
            ratio=(0.55, 1.8)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_data_loaders(root_dir="dataset", batch_size=16, val_split=0.15, seed=42):
    """
    Creates train and validation DataLoaders with balanced split.
    """
    train_transform, val_transform = get_transforms()

    # Load dataset with training transform
    full_dataset = CurrencyDataset(root_dir=root_dir, transform=train_transform)

    # Split sizes
    total_count = len(full_dataset)
    val_count = int(total_count * val_split)
    train_count = total_count - val_count

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [train_count, val_count], generator=generator)

    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(img_path).convert("RGB")
            if max(img.size) > 800:
                img.thumbnail((800, 800), Image.Resampling.BILINEAR)
            if self.transform:
                img = self.transform(img)
            return img, label

    val_dataset = TransformSubset(val_set, val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset

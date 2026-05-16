import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import RoomTypeDataset
from src.transforms import get_train_transforms, get_val_transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DEFAULT_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
        train_csv_path=None,
        val_csv_path=None,
        train_image_root=None,
        val_image_root=None,
        batch_size=32,
        num_workers=2,
        image_size=224,
        use_weighted_sampling=False,
        seed=None,
        pin_memory=False,
        persistent_workers=False,
):
    train_csv_path = train_csv_path or os.path.join(DEFAULT_PROCESSED_DIR, "train_df.csv")
    val_csv_path = val_csv_path or os.path.join(DEFAULT_PROCESSED_DIR, "val_df.csv")
    train_image_root = train_image_root or os.path.join(DEFAULT_RAW_DIR, "train_images")
    val_image_root = val_image_root or os.path.join(DEFAULT_RAW_DIR, "val_images")

    train_dataset = RoomTypeDataset(
        csv_path=train_csv_path,
        image_root=train_image_root,
        transform=get_train_transforms(image_size=image_size)
    )

    # validation без random augmentations
    val_dataset = RoomTypeDataset(
        csv_path=val_csv_path,
        image_root=val_image_root,
        transform=get_val_transforms(image_size=image_size)
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = partial(seed_worker, base_seed=seed)

    sampler = None
    if use_weighted_sampling:
        targets = train_dataset.df["result"].values

        class_counts = pd.Series(targets).value_counts().sort_index()
        class_weights = 1.0 / class_counts

        sample_weights = [class_weights[t] for t in targets]
        sample_weights = torch.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=use_weighted_sampling==False,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    # stable validation order
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_loader, val_loader


def create_test_dataloader(
        test_csv_path=None,
        test_image_root=None,
        batch_size=32,
        num_workers=2,
        image_size=224,
        filter_can_predict=True,
):
    test_csv_path = test_csv_path or os.path.join(DEFAULT_PROCESSED_DIR, "test_df.csv")
    test_image_root = test_image_root or os.path.join(DEFAULT_RAW_DIR, "test_images")

    test_dataset = RoomTypeDataset(
        csv_path=test_csv_path,
        image_root=test_image_root,
        transform=get_val_transforms(image_size=image_size),
        target_col=None,
        filter_can_predict=filter_can_predict
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader

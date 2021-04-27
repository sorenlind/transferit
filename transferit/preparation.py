"""Functions for preparing data."""

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

IMAGE_TYPES = ("jpg", "jpeg", "png")
T = TypeVar("T")


def create_class(
    source_folder: Path,
    dest_folder: Path,
    image_size: Tuple[int, int],
    n_max: Optional[int] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    files: List[Path] = []
    for image_type in IMAGE_TYPES:
        files.extend(source_folder.glob(f"**/*.{image_type}"))
    files = sorted(files)
    if n_max is not None:
        files = _take_n_random(files, n_max, seed)

    _resize_images(files, dest_folder, image_size)


def _take_n_random(items: List[T], n: int, seed: Optional[int]) -> List[T]:
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)
    items = items[:n]
    return items


def _resize_images(
    source_files: List[Path],
    dest_folder: Path,
    image_size: Tuple[int, int],
):
    for source_file in tqdm(source_files):
        target_file = dest_folder / source_file.name
        image = Image.open(source_file)
        image = image.resize(image_size, Image.BICUBIC)
        image.save(target_file)


def split_data(
    source_folder: Path,
    train_folder: Path,
    dev_folder: Path,
    seed: Optional[int],
    test_size: float,
):
    files = sorted(source_folder.glob("**/?*.*"))
    classes = [Path(f).parent.parts[-1] for f in files]
    class_counts = Counter(classes)
    print("Files per class:")
    for class_, n_images in class_counts.items():
        print(f" ◦  {n_images:6,} in class '{class_}'")

    train_files, dev_files = train_test_split(
        files,
        test_size=test_size,
        stratify=classes,
        random_state=seed,
    )

    print("")
    print("Files per set:")
    print(f" ◦  {len(train_files):6,} files in train set")
    print(f" ◦  {len(dev_files):6,} files in dev set")
    print("")
    print("")
    train_folder.mkdir(parents=True, exist_ok=False)
    dev_folder.mkdir(parents=True, exist_ok=False)

    print("Copying train files...")
    _copy_to_folder(source_folder, train_files, train_folder)
    print("Copying dev files...")
    _copy_to_folder(source_folder, dev_files, dev_folder)


def _copy_to_folder(source_folder, source_files, dest_folder):
    for source_file in tqdm(source_files):
        target_file = dest_folder / source_file.relative_to(source_folder)
        target_file.parent.mkdir(exist_ok=True, parents=False)
        shutil.copy(str(source_file), str(target_file))

"""Seeded report order of the generate-and-grade loop."""

import hashlib
import random

REFERENCE_SOURCE = "original_radiologist"


def model_seed(base_seed: int, model_path: str, model_type: str) -> int:
    """Per-model seed; the reference source (type "none") uses the base seed."""
    if model_type == "none":
        return base_seed
    return (base_seed + int(hashlib.sha256(model_path.encode()).hexdigest(), 16)) % 2**32


def generation_order(items: list, base_seed: int, model_path: str, model_type: str) -> list:
    """Shuffle a copy of `items` the way the generation loop did."""
    order = list(items)
    random.seed(model_seed(base_seed, model_path, model_type))
    random.shuffle(order)
    return order


def source_id(generation_model: dict) -> str:
    return REFERENCE_SOURCE if generation_model["type"] == "none" else generation_model["path"]

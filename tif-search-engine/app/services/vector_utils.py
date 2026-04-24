from __future__ import annotations

import numpy as np


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Encountered all-zero embedding vector.")
    return vector / norm

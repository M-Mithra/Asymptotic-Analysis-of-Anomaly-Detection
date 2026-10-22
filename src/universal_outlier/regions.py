"""Decision regions of the M = 3 binary rules in type space (the "cube").

With Bernoulli data each sequence is summarised by t_j = fraction of ones,
so a decision rule for three sequences is a partition of the unit cube
(t1, t2, t3) into "outlier 1/2/3" regions, plus

* the diagonal t1 = t2 = t3 ("no outlier": every hypothesis fits equally),
* tie sets where two hypotheses score the same ("ambiguous").

The rules are evaluated in the n -> infinity limit (``n=None``), i.e. as
functions of the types. By Sanov's theorem these regions determine the error
exponent (see :mod:`universal_outlier.theory`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import detectors as det_registry
from .decision import tie_mask
from .problem import Problem

NO_OUTLIER, AMBIGUOUS = 0, 4
CLASS_NAMES = {0: "No outlier", 1: "Outlier 1", 2: "Outlier 2", 3: "Outlier 3", 4: "Ambiguous"}
CLASS_COLORS = {0: "#000000", 1: "#d62728", 2: "#2ca02c", 3: "#1f77b4", 4: "#ff7f0e"}

_DEFAULT_PROBLEM = Problem.bernoulli(0.5, 0.5)


def grid_axis(step: float) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + step / 2, step), 10)


def classify(detector, t: np.ndarray, problem: Problem = _DEFAULT_PROBLEM) -> np.ndarray:
    """Class label (0..4) of each row of ``t`` (shape (N, 3), P(X=1) values)."""
    det = det_registry.get(detector) if isinstance(detector, str) else detector
    types = np.stack([1.0 - t, t], axis=-1)
    scores = det(types, None, problem)
    mask = tie_mask(scores)
    n_tied = mask.sum(axis=-1)
    labels = np.where(n_tied == 1, 1 + np.argmax(mask, axis=-1), AMBIGUOUS)
    equal = np.isclose(t[:, 0], t[:, 1]) & np.isclose(t[:, 1], t[:, 2])
    return np.where(equal, NO_OUTLIER, labels)


def classify_cube(detector, step: float = 0.025, problem: Problem = _DEFAULT_PROBLEM) -> pd.DataFrame:
    """Label every point of the (t1, t2, t3) grid with spacing ``step``."""
    ax = grid_axis(step)
    t1, t2, t3 = np.meshgrid(ax, ax, ax, indexing="ij")
    t = np.stack([t1.ravel(), t2.ravel(), t3.ravel()], axis=1)
    return pd.DataFrame({"t1": t[:, 0], "t2": t[:, 1], "t3": t[:, 2], "label": classify(detector, t, problem)})


def slice_labels(detector, axis: str, value: float, step: float = 0.005,
                 problem: Problem = _DEFAULT_PROBLEM):
    """Label image on the plane ``axis = value``; returns (u, v, image, names).

    ``image[i, j]`` is the label at (u[j], v[i]), ready for ``imshow``.
    """
    order = {"t1": (0, 1, 2), "t2": (1, 0, 2), "t3": (2, 0, 1)}[axis]
    names = [f"t{k + 1}" for k in order[1:]]
    ax = grid_axis(step)
    U, V = np.meshgrid(ax, ax, indexing="xy")
    t = np.empty((U.size, 3))
    t[:, order[0]] = value
    t[:, order[1]] = U.ravel()
    t[:, order[2]] = V.ravel()
    return ax, ax, classify(detector, t, problem).reshape(U.shape), names


def disagreement(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Fraction of grid points (off the diagonal) where two rules differ."""
    keep = df_a.label != NO_OUTLIER
    return float((df_a.label[keep] != df_b.label[keep]).mean())

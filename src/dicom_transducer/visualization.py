# ruff: noqa: F722
import matplotlib.pyplot as plt

from typing import Any

from numpy import ndarray

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from jaxtyping import Float


def create_gridplot(
    datamapping: dict[Any, Float[ndarray, 'D H W']],
    zindex: int | None,
    ncols: int = 3,
    figsize_multiplier: float = 5.0
) -> tuple[Figure, Axes]:
    nrows: int = (len(datamapping) + ncols - 1) // ncols
    figsize: tuple[float, float] = (ncols * figsize_multiplier, nrows * figsize_multiplier)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    axes: list[Axes]
    
    for ax, (key, value) in zip(axes.flat, datamapping.items()):
        ax: Axes
        zindex = zindex if zindex is not None else value.shape[0] // 2
        ax.imshow(value[zindex], cmap='gray')
        ax.set_title(str(key))
        ax.axis('off')

    return (fig, axes)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from tqdm.auto import tqdm

from .types import DicomTree, ActualizedDicomTree, DicomVolume


def _dataset_to_dict(ds: pydicom.Dataset) -> dict[str, Any]:
    """Convert a pydicom Dataset to a plain Python dict, excluding pixel data."""
    result: dict[str, Any] = {}
    for elem in ds:
        if elem.tag.group == 0x7FE0:  # Pixel Data group
            continue
        key = elem.keyword or str(elem.tag)
        if elem.VR == 'SQ':
            result[key] = [_dataset_to_dict(item) for item in elem.value]
        elif isinstance(elem.value, pydicom.valuerep.PersonName):
            result[key] = str(elem.value)
        elif isinstance(elem.value, pydicom.multival.MultiValue):
            result[key] = [v if not isinstance(v, pydicom.uid.UID) else str(v)
                           for v in elem.value]
        elif isinstance(elem.value, pydicom.uid.UID):
            result[key] = str(elem.value)
        else:
            result[key] = elem.value
    return result


def _collect_leaves(
    tree: DicomTree,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], list[Path]]]:
    """Collect all leaf entries from a DicomTree with their key paths."""
    leaves: list[tuple[tuple[str, ...], list[Path]]] = []
    for key, value in tree.items():
        key_path = (*prefix, key)
        if isinstance(value, list):
            leaves.append((key_path, value))
        else:
            leaves.extend(_collect_leaves(value, key_path))
    return leaves


def _reassemble_tree(
    results: dict[tuple[str, ...], DicomVolume],
) -> ActualizedDicomTree:
    """Reassemble flat results into a nested ActualizedDicomTree."""
    root: ActualizedDicomTree = {}
    for key_path, volume in results.items():
        node = root
        for key in key_path[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]  # type: ignore[assignment]
        node[key_path[-1]] = volume
    return root


def actualize(
    tree: DicomTree,
    *,
    max_workers: int | None = None,
) -> ActualizedDicomTree:
    """Load all DICOM directories in a tree into DicomVolume objects.

    Traverses the nested *tree* and loads each leaf (``list[Path]``) into a
    :class:`DicomVolume` by reading the DICOM files and applying the modality
    LUT.  Leaf directories are loaded in parallel using a thread pool.

    Args:
        tree: Parsed DICOM tree from :func:`parse_dicom_tree`.
        max_workers: Maximum number of threads.  ``None`` lets the
            :class:`~concurrent.futures.ThreadPoolExecutor` choose a default
            based on the number of CPUs.
    """
    leaves = _collect_leaves(tree)

    results: dict[tuple[str, ...], DicomVolume] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(load_dicom_directory, paths): key_path
            for key_path, paths in leaves
        }
        with tqdm(total=len(leaves), desc="Directories", unit="dir") as bar:
            for future in as_completed(future_to_key):
                key_path = future_to_key[future]
                results[key_path] = future.result()
                bar.set_postfix_str(key_path[-1], refresh=False)
                bar.update(1)

    return _reassemble_tree(results)


def load_dicom_directory(
    paths: list[Path],
    *,
    verbose: bool = False,
) -> DicomVolume:
    """Load DICOM slices into a 3D volume with the modality LUT applied.

    Reads each DICOM file in *paths* (expected to be pre-sorted by
    :func:`parse_dicom_directory`), applies the modality LUT
    (RescaleSlope / RescaleIntercept) to convert raw stored values to
    the rescaled representation (e.g. Hounsfield Units for CT), and
    stacks the slices into a single 3D NumPy array.

    Args:
        paths: Sorted list of ``.dcm`` file paths (as returned by
            :func:`parse_dicom_directory`).
        verbose: If ``True``, print progress information.

    Returns:
        A ``DicomVolume`` object where *volume* is a float64 array
        of shape ``(slices, rows, cols)`` and *metadata* is a dict of
        DICOM header fields from the first slice.
    """
    datasets = [pydicom.dcmread(p) for p in paths]

    slices = [
        pydicom.pixels.apply_modality_lut(ds.pixel_array, ds)
        for ds in datasets
    ]
    volume = np.stack(slices, axis=0)
    metadata = _dataset_to_dict(datasets[0])

    if verbose:
        print(
            f"Loaded volume: shape={volume.shape}, dtype={volume.dtype}, "
            f"range=[{volume.min():.1f}, {volume.max():.1f}]"
        )

    return DicomVolume(volume=volume, metadata=metadata)

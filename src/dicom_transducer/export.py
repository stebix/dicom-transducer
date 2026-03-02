import json
from pathlib import Path

import numpy as np
import zarr

from .dicom import ActualizedDicomTree, DicomVolume
from .naming import generate_unique_names


def _sanitize_for_json(obj: object) -> object:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_zarr(
    volume: DicomVolume,
    path: str | Path,
    *,
    force_write: bool = False,
) -> None:
    """Export a DicomVolume to a zarr v3 store.

    Creates the structure:
        <path>/raw/full  — array with volume data and metadata as attributes

    Raises:
        FileExistsError: If *path* already exists and *force_write* is False.
    """
    path = Path(path)
    if path.exists() and not force_write:
        raise FileExistsError(
            f"'{path}' already exists. Pass force_write=True to overwrite."
        )
    root = zarr.open_group(path, mode="w", zarr_format=3)
    raw = root.create_group("raw")
    arr = raw.create_array("full", data=volume.volume)
    arr.update_attributes(_sanitize_for_json(volume.metadata))


def flatten_to_volumes(tree: ActualizedDicomTree) -> dict[str, DicomVolume]:
    """Flatten a 2-level ActualizedDicomTree to ``{integer_id: DicomVolume}``.

    Expects the structure produced by :func:`actualize` on a directory laid out
    as ``{integer_dir: {series_name: DicomVolume}}``.  Each volume's metadata
    is augmented with ``"source_directory"`` and ``"series_directory"`` keys.
    The original volumes are not mutated.

    Raises:
        ValueError: If any top-level entry is not a single-child dict
            containing a :class:`DicomVolume`.
    """
    result: dict[str, DicomVolume] = {}
    for parent_key, subtree in tree.items():
        if isinstance(subtree, DicomVolume):
            raise ValueError(
                f"Expected nested dict under '{parent_key}', got DicomVolume directly"
            )
        if len(subtree) != 1:
            raise ValueError(
                f"Expected exactly 1 series under '{parent_key}', got {len(subtree)}"
            )
        series_name, volume = next(iter(subtree.items()))
        if not isinstance(volume, DicomVolume):
            raise ValueError(
                f"Expected DicomVolume under '{parent_key}/{series_name}', "
                f"got nested tree"
            )
        augmented_metadata = {
            **volume.metadata,
            "source_directory": parent_key,
            "series_directory": series_name,
        }
        result[parent_key] = DicomVolume(
            volume=volume.volume, metadata=augmented_metadata
        )
    return result


def export_zarr_collection(
    volumes: dict[str, DicomVolume],
    output_dir: str | Path,
    *,
    seed: int | None = None,
    force_write: bool = False,
) -> dict[str, str]:
    """Export a collection of DicomVolumes to individual zarr stores.

    For each volume a human-readable name is generated (e.g.
    ``"gallivanting-groundhog"``) and the volume is written to
    ``output_dir/{name}.zarr/`` via :func:`export_zarr`.

    A ``manifest.json`` mapping ``{integer_id: generated_name}`` is written
    alongside the zarr stores for future reference.

    All target paths are checked for conflicts *before* any data is written,
    so a partial write cannot occur when *force_write* is False.

    Args:
        volumes: Flat mapping of integer directory ids to DicomVolumes
            (as returned by :func:`flatten_to_volumes`).
        output_dir: Parent directory for all outputs.
        seed: Optional RNG seed for reproducible name generation.
        force_write: If False (default), raise on any existing output path.

    Returns:
        The ``{integer_id: generated_name}`` mapping that was persisted.

    Raises:
        FileExistsError: If *output_dir* already contains conflicting paths
            and *force_write* is False.
    """
    import random

    output_dir = Path(output_dir)

    rng = random.Random(seed)
    sorted_keys = sorted(volumes, key=int)
    names = generate_unique_names(len(sorted_keys), rng=rng)
    mapping: dict[str, str] = dict(zip(sorted_keys, names))

    # Pre-flight: check all targets before writing anything.
    if not force_write:
        manifest_path = output_dir / "manifest.json"
        conflicts = [
            str(output_dir / f"{name}.zarr")
            for name in names
            if (output_dir / f"{name}.zarr").exists()
        ]
        if manifest_path.exists():
            conflicts.append(str(manifest_path))
        if conflicts:
            raise FileExistsError(
                f"Refusing to overwrite existing paths:\n"
                + "\n".join(f"  - {c}" for c in conflicts)
                + "\nPass force_write=True to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted_keys:
        export_zarr(
            volumes[key],
            output_dir / f"{mapping[key]}.zarr",
            force_write=force_write,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(mapping, indent=2))

    return mapping

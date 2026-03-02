import warnings
from pathlib import Path

from .types import DicomTree


def parse_dicom_tree(path: Path, *, verbose: bool = False) -> DicomTree:
    """Recursively walk a directory tree and parse DICOM leaf directories.

    Each subdirectory of *path* is classified as:
    - **leaf** (contains only ``.dcm`` files) — parsed via
      :func:`parse_dicom_directory`.
    - **intermediate** (contains only subdirectories) — recursed into.
    - **mixed** (``.dcm`` files *and* subdirectories) — raises
      :class:`ValueError`.
    - **empty / irrelevant** — silently skipped.

    Invalid DICOM leaves (e.g. non-contiguous slices) are skipped with a
    warning instead of raising.

    Returns:
        A nested dict whose leaf values are ``list[Path]`` and intermediate
        values are further ``DicomTree`` dicts.

    Raises:
        FileNotFoundError: If *path* is not an existing directory.
        ValueError: If any subdirectory mixes ``.dcm`` files and
            subdirectories.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"'{path}' is not an existing directory")

    result: DicomTree = {}

    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir():
            continue

        dcm_files = [
            item for item in subdir.iterdir()
            if item.is_file() and item.suffix.lower() == '.dcm'
        ]
        subdirs = [
            item for item in subdir.iterdir()
            if item.is_dir()
        ]

        has_dcm = len(dcm_files) > 0
        has_subdirs = len(subdirs) > 0

        if has_dcm and has_subdirs:
            raise ValueError(
                f"Mixed directory '{subdir}' contains both .dcm files and subdirectories"
            )

        if has_dcm:
            try:
                result[subdir.name] = parse_dicom_directory(subdir, verbose=verbose)
            except ValueError as exc:
                warnings.warn(f"Skipping '{subdir.name}': {exc}")
        elif has_subdirs:
            result[subdir.name] = parse_dicom_tree(subdir, verbose=verbose)

    return result


def _are_consecutive(numbers: list[int]) -> bool:
    if len(numbers) < 2:
        return True
    sorted_numbers = sorted(numbers)
    return all(
        b - a == 1 for a, b in zip(sorted_numbers, sorted_numbers[1:])
    )


def parse_dicom_directory(
    path: Path,
    *,
    verbose: bool = False
) -> list[Path]:
    """Parse a directory of DICOM files and return a sorted list of paths.

    Validates that the directory contains DICOM files (.dcm) whose filename
    stems are integer slice numbers forming a contiguous sequence.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a directory.
        ValueError: If the directory contains no .dcm files, if any filename
            stem is not a valid integer, or if the slice numbers are not
            contiguous.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"'{path}' is not an existing directory")

    dcm_files = sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() == '.dcm'),
        key=lambda p: int(p.stem),
    )

    if not dcm_files:
        raise ValueError(f"No .dcm files found in '{path}'")

    try:
        slice_numbers = [int(p.stem) for p in dcm_files]
    except ValueError as exc:
        raise ValueError(
            f"Non-integer DICOM filename stem in '{path}'"
        ) from exc

    if not _are_consecutive(slice_numbers):
        missing = {
            i for i in range(min(slice_numbers), max(slice_numbers) + 1)
        } - set(slice_numbers)
        raise ValueError(
            f"DICOM slice numbers in '{path}' are not contiguous. "
            f"Missing slices: {sorted(missing)}"
        )

    if verbose:
        print(f"Found {len(dcm_files)} contiguous DICOM slices in '{path}'")

    return dcm_files

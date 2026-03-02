"""Command-line interface for dicom-transducer."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dicom-transducer",
        description="Transduce DICOM directory trees into zarr stores.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root DICOM directory (integer subdirs, each with one series).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for zarr stores and manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible name generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel threads for loading directories (default: CPU count).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    from .dicom import actualize, parse_dicom_tree
    from .export import export_zarr_collection, flatten_to_volumes

    print(f"Parsing DICOM tree at {input_dir} ...")
    tree = parse_dicom_tree(input_dir, verbose=args.verbose)

    n_series = len(tree)
    if n_series == 0:
        print("No DICOM series found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {n_series} series.")

    print("Loading volumes ...")
    actualized = actualize(tree, max_workers=args.workers)

    volumes = flatten_to_volumes(actualized)
    if args.verbose:
        for key, vol in sorted(volumes.items(), key=lambda kv: int(kv[0])):
            print(f"  {key}: shape={vol.volume.shape}")

    print(f"Exporting to {output_dir} ...")
    try:
        mapping = export_zarr_collection(
            volumes,
            output_dir,
            seed=args.seed,
            force_write=args.force,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print("Manifest:")
    for key in sorted(mapping, key=int):
        print(f"  {key} -> {mapping[key]}")

    print("Done.")

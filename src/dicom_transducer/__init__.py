from .dicom import (
    DicomTree,
    ActualizedDicomTree,
    DicomVolume,
    parse_dicom_tree,
    parse_dicom_directory,
    actualize,
    load_dicom_directory,
)
from .export import export_zarr, export_zarr_collection, flatten_to_volumes
from .naming import generate_name, generate_unique_names


def main() -> None:
    from .cli import main as _cli_main
    _cli_main()


if __name__ == '__main__':
    main()

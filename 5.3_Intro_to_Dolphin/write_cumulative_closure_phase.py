from pathlib import Path


def write_cumulative_closure_phase(
    interferograms_dir: Path,
    wavelength: float,
    output_dir: Path | None = None,
    glob_pattern: str = "closure_phase_*.tif",
) -> list[Path]:
    """Sum closure phase rasters and convert to displacement in meters.

    Cumulatively sums per-date ``closure_phase_*.tif`` rasters from dolphin's
    ``interferograms/`` folder and converts to displacement via
    ``displacement = -wavelength * phase / (4 pi)`` so positive values indicate
    apparent motion toward the sensor.

    Parameters
    ----------
    interferograms_dir
        Directory holding ``closure_phase_<date>.tif`` rasters (dolphin's
        ``interferograms/`` folder).
    wavelength
        Radar wavelength in meters.
    output_dir
        Destination directory for ``cumulative_closure_phase_<date>.tif`` files.
        Defaults to ``interferograms_dir`` (matching the dolphin script).
    glob_pattern
        Glob pattern for closure phase files.

    Returns
    -------
    list[Path]
        Written cumulative closure phase files.  Empty list if no input
        files are found.

    """
    import os

    import numpy as np
    from dolphin import io
    from opera_utils import get_dates

    interferograms_dir = Path(interferograms_dir)
    input_files = sorted(interferograms_dir.glob(glob_pattern))
    if not input_files:
        print(
            f"No closure phase files in {interferograms_dir}; skipping cumulative closure phase."
        )
        return []

    out_dir = Path(output_dir) if output_dir is not None else interferograms_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _out_path(fin: str | os.PathLike) -> Path:
        date_str = get_dates(Path(fin))[1].strftime("%Y%m%d")
        return out_dir / f"cumulative_closure_phase_{date_str}.tif"

    expected = [_out_path(f) for f in input_files]
    if all(p.exists() for p in expected):
        print("Skipping cumulative closure phase (exists)")
        return expected

    reader = io.RasterStackReader.from_file_list(input_files)
    running_sum = np.zeros(reader.shape[1:], dtype="float64")
    written: list[Path] = []
    for idx, fin in enumerate(reader.file_list):
        block = np.ma.filled(reader[idx, :, :], 0)
        running_sum += np.asarray(block).squeeze().astype("float64")
        # -wavelength/(4 pi) converts phase -> meters with positive = toward sensor.
        displacement = (running_sum * wavelength / -4.0 / np.pi).astype("float32")
        fout = _out_path(fin)
        io.write_arr(
            arr=displacement,
            output_name=fout,
            like_filename=fin,
            options=io.EXTRA_COMPRESSED_TIFF_OPTIONS,
        )
        written.append(fout)
    print(f"Wrote {len(written)} cumulative closure phase files to {out_dir}")
    return written

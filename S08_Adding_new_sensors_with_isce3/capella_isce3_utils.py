"""Support code for the "Adding new sensors with ISCE3" notebook.

These are the ISCE3 coregistration steps, ported from ``docs/examples/`` in
https://github.com/capellaspace/capella-reader, so the notebook can stay focused
on the sensor interface rather than on plumbing.

References
----------
.. [1] Capella Space SAR Products Format Specification v1.8.
   https://support.capellaspace.com/hubfs/Capella_Space_SAR_Products_Format_Specification_v1.8.pdf
.. [2] Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).
   Efficient subpixel image registration algorithms. Optics Letters, 33(2),
   156-158. https://doi.org/10.1364/OL.33.000156

"""

from __future__ import annotations

import time
from os import fsdecode
from pathlib import Path

import capella_reader.adapters.isce3  # noqa: F401  (registers the isce3 adapter)
import isce3
import numpy as np
from capella_reader import CapellaSLC, adapters
from osgeo import gdal
from skimage.registration import phase_cross_correlation

gdal.UseExceptions()

__all__ = [
    "MEXICO_CITY_SLC_URLS",
    "bulk_offset",
    "coregister_pair",
    "correlate_grid",
    "form_interferogram",
    "multilook",
    "resample_slc",
    "run_geo2rdr",
    "run_rdr2geo",
    "write_envi_header",
]


# Capella Mexico City stripmap SLCs, June-August 2024: 18 acquisitions from the
# C14 satellite, HH polarization, on a 3-day repeat.
# https://www.capellaspace.com/earth-observation-solutions/capella-open-data/
_BASE = "https://capella-open-data.s3.amazonaws.com/data"
_SCENE_IDS = [
    ("2024/6/26", "CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055"),
    ("2024/6/29", "CAPELLA_C14_SM_SLC_HH_20240629134910_20240629134915"),
    ("2024/7/2", "CAPELLA_C14_SM_SLC_HH_20240702123719_20240702123723"),
    ("2024/7/5", "CAPELLA_C14_SM_SLC_HH_20240705112536_20240705112541"),
    ("2024/7/8", "CAPELLA_C14_SM_SLC_HH_20240708101348_20240708101353"),
    ("2024/7/11", "CAPELLA_C14_SM_SLC_HH_20240711090203_20240711090207"),
    ("2024/7/14", "CAPELLA_C14_SM_SLC_HH_20240714075018_20240714075023"),
    ("2024/7/17", "CAPELLA_C14_SM_SLC_HH_20240717063832_20240717063836"),
    ("2024/7/20", "CAPELLA_C14_SM_SLC_HH_20240720052647_20240720052651"),
    ("2024/7/23", "CAPELLA_C14_SM_SLC_HH_20240723041503_20240723041508"),
    ("2024/7/26", "CAPELLA_C14_SM_SLC_HH_20240726030317_20240726030321"),
    ("2024/7/29", "CAPELLA_C14_SM_SLC_HH_20240729015127_20240729015132"),
    ("2024/8/1", "CAPELLA_C14_SM_SLC_HH_20240801003948_20240801003953"),
    ("2024/8/3", "CAPELLA_C14_SM_SLC_HH_20240803232806_20240803232810"),
    ("2024/8/6", "CAPELLA_C14_SM_SLC_HH_20240806221608_20240806221613"),
    ("2024/8/9", "CAPELLA_C14_SM_SLC_HH_20240809210429_20240809210433"),
    ("2024/8/12", "CAPELLA_C14_SM_SLC_HH_20240812195242_20240812195246"),
    ("2024/8/15", "CAPELLA_C14_SM_SLC_HH_20240815184057_20240815184102"),
]
MEXICO_CITY_SLC_URLS = [f"{_BASE}/{d}/{n}/{n}.tif" for d, n in _SCENE_IDS]


# ---------------------------------------------------------------------------
# ISCE3 geometry and resampling
# ---------------------------------------------------------------------------


def open_slc_isce3(
    slc_file: Path,
) -> tuple[CapellaSLC, isce3.product.RadarGridParameters, isce3.core.Orbit, object]:
    """Open a Capella SLC and return ``(slc, radar_grid, orbit, ellipsoid)``."""
    import warnings

    slc = CapellaSLC.from_file(slc_file)
    radar_grid = adapters.isce3.get_radar_grid(slc)
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        orbit = adapters.isce3.get_orbit(slc)
    ellipsoid = isce3.core.make_projection(4326).ellipsoid
    return slc, radar_grid, orbit, ellipsoid


def run_rdr2geo(slc_file: Path, dem_file: Path, output_dir: Path) -> Path:
    """Map every reference pixel to lon/lat/height; return the 3-band VRT.

    This is the "where on the ground is each radar pixel" half of geometric
    coregistration.
    """
    geom_dir = output_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    out_vrt = geom_dir / "geometry.vrt"
    if out_vrt.exists():
        return out_vrt

    _, radar_grid, orbit, ellipsoid = open_slc_isce3(slc_file)
    rdr2geo = isce3.geometry.Rdr2Geo(
        radar_grid,
        orbit,
        ellipsoid,
        isce3.core.LUT2d(),
        threshold=1e-8,
        numiter=20,
        extraiter=10,
        lines_per_block=1024,
    )

    def _layer(name: str) -> isce3.io.Raster:
        return isce3.io.Raster(
            fsdecode(geom_dir / f"{name}.tif"),
            radar_grid.width,
            radar_grid.length,
            1,
            gdal.GDT_Float64,
            "GTiff",
        )

    x_raster, y_raster, z_raster = _layer("x"), _layer("y"), _layer("z")
    rdr2geo.topo(
        isce3.io.Raster(fsdecode(dem_file)),
        x_raster=x_raster,
        y_raster=y_raster,
        height_raster=z_raster,
    )
    stack = isce3.io.Raster(fsdecode(out_vrt), [x_raster, y_raster, z_raster])
    stack.set_epsg(rdr2geo.epsg_out)
    del stack, x_raster, y_raster, z_raster
    return out_vrt


def run_geo2rdr(
    sec_file: Path, geometry_vrt: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Map the reference ground positions back into the secondary radar grid.

    Returns the ``(range.off, azimuth.off)`` pair: for each reference pixel,
    where to sample the secondary.
    """
    g2r_dir = output_dir / "geo2rdr"
    g2r_dir.mkdir(parents=True, exist_ok=True)

    _, radar_grid, orbit, ellipsoid = open_slc_isce3(sec_file)
    geo2rdr = isce3.geometry.Geo2Rdr(
        radar_grid,
        orbit,
        ellipsoid,
        isce3.core.LUT2d(),  # Capella SLCs are already on a zero-Doppler grid
        1e-8,
        20,
        1024,
    )
    geo2rdr.geo2rdr(isce3.io.Raster(fsdecode(geometry_vrt)), fsdecode(g2r_dir))
    return g2r_dir / "range.off", g2r_dir / "azimuth.off"


def resample_slc(
    ref_file: Path,
    sec_file: Path,
    rg_off_path: Path,
    az_off_path: Path,
    output_file: Path,
) -> Path:
    """Resample the secondary SLC onto the reference radar grid.

    ``flatten=True`` removes the geometric (range difference) phase, leaving
    the deformation and atmospheric terms behind.
    """
    ref_slc = CapellaSLC.from_file(ref_file)
    sec_slc = CapellaSLC.from_file(sec_file)

    resamp = isce3.image.ResampSlc(
        adapters.isce3.get_radar_grid(sec_slc),
        adapters.isce3.get_doppler_lut2d(sec_slc),
        isce3.core.Poly2d(np.array([0.0])),  # no azimuth carrier
        isce3.core.Poly2d(np.array([0.0])),  # no range carrier
        0.0j,
        adapters.isce3.get_radar_grid(ref_slc),
    )
    resamp.lines_per_tile = 1024

    rg_off_r = isce3.io.Raster(fsdecode(rg_off_path))
    az_off_r = isce3.io.Raster(fsdecode(az_off_path))
    in_raster = isce3.io.Raster(fsdecode(sec_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_raster = isce3.io.Raster(
        fsdecode(output_file),
        rg_off_r.width,
        rg_off_r.length,
        1,
        gdal.GDT_CFloat32,
        "GTiff",
    )
    resamp.resamp(in_raster, out_raster, rg_off_r, az_off_r, flatten=True)
    del in_raster, out_raster, rg_off_r, az_off_r

    # Carry the Capella metadata onto the output so it stays readable by
    # capella-reader (the resampled product now lives on the reference grid,
    # but the sensor metadata is still the secondary's).
    ds_in = gdal.Open(str(sec_file))
    ds_out = gdal.Open(str(output_file), gdal.GA_Update)
    ds_out.SetMetadataItem(
        "TIFFTAG_IMAGEDESCRIPTION", ds_in.GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION")
    )
    ds_in = ds_out = None
    return output_file


def write_envi_header(
    file_path: Path, lines: int, samples: int, dtype: np.dtype
) -> None:
    """Write a minimal ENVI ``.hdr`` sidecar so ISCE3 can open a raw offset file."""
    envi_dtypes = {np.dtype("float32"): 4, np.dtype("float64"): 5}
    Path(str(file_path) + ".hdr").write_text(
        "ENVI\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        f"data type = {envi_dtypes[dtype]}\n"
        "interleave = bsq\n"
    )


# ---------------------------------------------------------------------------
# Cross-correlation refinement
# ---------------------------------------------------------------------------


def correlate_grid(
    ref_file: Path,
    sec_file: Path,
    *,
    chip_size: tuple[int, int] = (256, 256),
    n_chips: tuple[int, int] = (12, 12),
    upsample_factor: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel offsets on a regular grid of amplitude chips.

    Chips are read one at a time through GDAL rather than loading the images,
    which keeps this usable on a full 200-megapixel stripmap frame.

    Parameters
    ----------
    ref_file, sec_file : Path
        Reference and (coarsely resampled) secondary SLCs, on the same grid.
    chip_size : tuple[int, int]
        Correlation window size in (rows, cols).
    n_chips : tuple[int, int]
        How many chips to place down each axis, spread evenly over the frame.
    upsample_factor : int
        Sub-pixel refinement factor for ``phase_cross_correlation``.

    Returns
    -------
    az_off, rg_off : np.ndarray
        Per-chip azimuth (row) and range (col) offsets, in the convention
        ISCE3's resampler wants: the value to ADD to a reference index to get
        the matching secondary index.
    error : np.ndarray
        Normalized RMS error per chip (0 is a perfect match).

    """
    ref_ds = gdal.Open(str(ref_file))
    sec_ds = gdal.Open(str(sec_file))
    nrows, ncols = ref_ds.RasterYSize, ref_ds.RasterXSize
    assert (sec_ds.RasterYSize, sec_ds.RasterXSize) == (nrows, ncols), "grid mismatch"

    ch, cw = chip_size
    row_starts = np.linspace(0, nrows - ch, n_chips[0]).astype(int)
    col_starts = np.linspace(0, ncols - cw, n_chips[1]).astype(int)

    az_off, rg_off, err = (
        np.full(len(row_starts) * len(col_starts), np.nan) for _ in range(3)
    )
    i = 0
    for r0 in row_starts:
        for c0 in col_starts:
            ref = np.abs(ref_ds.ReadAsArray(int(c0), int(r0), cw, ch))
            sec = np.abs(sec_ds.ReadAsArray(int(c0), int(r0), cw, ch))
            # A chip that is all zeros (outside the resampled overlap) has no
            # correlation peak to find.
            if not (ref.any() and sec.any()):
                i += 1
                continue
            # normalization=None gives plain cross-correlation, which Guizar
            # et al. recommend for noisy amplitude imagery.
            shift, error, _ = phase_cross_correlation(
                ref,
                sec,
                upsample_factor=upsample_factor,
                normalization=None,
            )
            # phase_cross_correlation returns the shift that registers the
            # moving image onto the reference; the resampler wants the inverse.
            az_off[i], rg_off[i], err[i] = -shift[0], -shift[1], error
            i += 1
    return az_off, rg_off, err


def bulk_offset(
    az_off: np.ndarray, rg_off: np.ndarray, *, n_mad: float = 3.0
) -> tuple[float, float, np.ndarray]:
    """Robust constant (azimuth, range) shift from per-chip offsets.

    Iteratively drops chips more than `n_mad` median-absolute-deviations from
    the median in either axis, then returns the median of the survivors.

    Returns
    -------
    az_median, rg_median : float
        Constant offsets in pixels.
    inliers : np.ndarray
        Boolean mask of the chips that were used.

    """
    inliers = np.isfinite(az_off) & np.isfinite(rg_off)
    assert inliers.any(), "No finite chip offsets"
    for _ in range(3):
        az_med, rg_med = np.median(az_off[inliers]), np.median(rg_off[inliers])
        az_mad = 1.4826 * np.median(np.abs(az_off[inliers] - az_med)) + 1e-6
        rg_mad = 1.4826 * np.median(np.abs(rg_off[inliers] - rg_med)) + 1e-6
        new_inliers = (
            np.isfinite(az_off)
            & np.isfinite(rg_off)
            & (np.abs(az_off - az_med) < n_mad * az_mad)
            & (np.abs(rg_off - rg_med) < n_mad * rg_mad)
        )
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers
    return float(np.median(az_off[inliers])), float(np.median(rg_off[inliers])), inliers


# ---------------------------------------------------------------------------
# The whole pair, end to end
# ---------------------------------------------------------------------------


def coregister_pair(
    ref_file: Path,
    sec_file: Path,
    geometry_vrt: Path,
    output_dir: Path,
    output_file: Path,
    *,
    verbose: bool = True,
) -> Path:
    """Coregister one secondary SLC to the reference, geometry plus refinement.

    Assumes ``run_rdr2geo`` has already been run on the reference (its output
    only depends on the reference and the DEM, so it is reused across the
    stack).

    Parameters
    ----------
    ref_file, sec_file : Path
        Reference and secondary Capella SLCs.
    geometry_vrt : Path
        3-band lon/lat/height VRT from `run_rdr2geo`.
    output_dir : Path
        Scratch directory for offsets and the intermediate coarse resample.
    output_file : Path
        Where to write the final coregistered SLC.
    verbose : bool
        Print per-step timing and the fitted bulk offset.

    Returns
    -------
    Path
        `output_file`.

    """
    if output_file.exists():
        return output_file
    t0 = time.time()

    rg_off, az_off = run_geo2rdr(sec_file, geometry_vrt, output_dir)
    coarse_file = output_dir / "coarse_resampled.tif"
    coarse_file.unlink(missing_ok=True)
    resample_slc(ref_file, sec_file, rg_off, az_off, coarse_file)

    az_chips, rg_chips, _ = correlate_grid(ref_file, coarse_file)
    az_med, rg_med, inliers = bulk_offset(az_chips, rg_chips)
    if verbose:
        print(
            f"  bulk offset az={az_med:+.3f} rg={rg_med:+.3f} px"
            f"  ({inliers.sum()}/{len(az_chips)} chips)"
        )

    # geo2rdr writes float64 rasters; ISCE3's resampler wants the same, so the
    # combined offsets are written as float64 with an ENVI sidecar.
    combined_dir = output_dir / "combined_offsets"
    combined_dir.mkdir(parents=True, exist_ok=True)
    rg_raster = isce3.io.Raster(fsdecode(rg_off))
    nrows, ncols = rg_raster.length, rg_raster.width
    del rg_raster

    for coarse_path, shift, name in (
        (rg_off, rg_med, "range.off"),
        (az_off, az_med, "azimuth.off"),
    ):
        coarse = np.memmap(coarse_path, dtype=np.float64, mode="r", shape=(nrows, ncols))
        combined = np.memmap(
            combined_dir / name, mode="w+", dtype=np.float64, shape=(nrows, ncols)
        )
        combined[:] = coarse + shift
        combined.flush()
        del coarse, combined
        write_envi_header(combined_dir / name, nrows, ncols, np.dtype("float64"))

    resample_slc(
        ref_file,
        sec_file,
        combined_dir / "range.off",
        combined_dir / "azimuth.off",
        output_file,
    )
    if verbose:
        print(f"  {output_file.name} in {time.time() - t0:.1f} s")
    return output_file


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def multilook(image: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    """Average `image` in blocks of ``(row_looks, col_looks)``, trimming the edge."""
    row_looks, col_looks = looks
    nrows = (image.shape[0] // row_looks) * row_looks
    ncols = (image.shape[1] // col_looks) * col_looks
    trimmed = image[:nrows, :ncols]
    return trimmed.reshape(
        nrows // row_looks, row_looks, ncols // col_looks, col_looks
    ).mean(axis=(1, 3))


def form_interferogram(
    ref_file: Path,
    sec_file: Path,
    looks: tuple[int, int] = (16, 16),
    block_rows: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the multilooked interferogram and its coherence.

    Reads in row blocks so a full 200-megapixel frame does not have to fit in
    memory twice over.

    Parameters
    ----------
    ref_file, sec_file : Path
        Two SLCs on the same grid.
    looks : tuple[int, int]
        Multilook factors as (row_looks, col_looks).
    block_rows : int
        Approximate number of input rows to read at a time.

    Returns
    -------
    ifg : np.ndarray
        Complex ``<ref * conj(sec)>``.
    coherence : np.ndarray
        ``|<ref conj(sec)>| / sqrt(<|ref|^2> <|sec|^2>)``.

    """
    ref_ds = gdal.Open(str(ref_file))
    sec_ds = gdal.Open(str(sec_file))
    nrows, ncols = ref_ds.RasterYSize, ref_ds.RasterXSize
    assert (sec_ds.RasterYSize, sec_ds.RasterXSize) == (nrows, ncols), "grid mismatch"

    row_looks, col_looks = looks
    out_rows, out_cols = nrows // row_looks, ncols // col_looks
    read_cols = out_cols * col_looks

    ifg = np.zeros((out_rows, out_cols), dtype=np.complex64)
    power = np.zeros((out_rows, out_cols), dtype=np.float64)

    step = max(row_looks, (block_rows // row_looks) * row_looks)
    for r0 in range(0, out_rows * row_looks, step):
        n_read = min(step, out_rows * row_looks - r0)
        ref = ref_ds.ReadAsArray(0, r0, read_cols, n_read)
        sec = sec_ds.ReadAsArray(0, r0, read_cols, n_read)
        out_slice = slice(r0 // row_looks, r0 // row_looks + n_read // row_looks)
        ifg[out_slice] = multilook(ref * np.conj(sec), looks)
        power[out_slice] = multilook(np.abs(ref) ** 2.0, looks) * multilook(
            np.abs(sec) ** 2.0, looks
        )

    # Coherence is undefined where either image has no signal (the resampler
    # writes zeros outside the valid overlap).
    coherence = np.full(ifg.shape, np.nan)
    valid = power > 0
    coherence[valid] = np.abs(ifg[valid]) / np.sqrt(power[valid])
    return ifg, coherence

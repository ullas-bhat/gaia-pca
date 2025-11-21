import argparse

import dask.array as da
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="PCA analysis on Gaia DR3 data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

pca_args = parser.add_argument_group("PCA options")
pca_args.add_argument(
    "-st",
    "--snr-thresholds",
    nargs="+",
    type=float,
    default=[0, 20, 30, 40, 50, 75, 100],
    metavar="SNR",
    help="The SNR thresholds to fit the PCs to.",
)
pca_args.add_argument(
    "-npc", "--num-pcs", type=int, default=5, metavar="N", help="Number of PCs to fit."
)

pca_args.add_argument(
    "-wr",
    "--wavelength-range",
    nargs=2,
    type=int,
    metavar=("MIN_IDX", "MAX_IDX"),
    default=[2, -2],
    help="The wavelength range (as indices) to use for the PCA analysis. By default, the first two and last two wavelengths are excluded.",
)

data_args = parser.add_argument_group("Data options")
data_args.add_argument(
    "-i",
    "--input-name",
    type=str,
    default="./gaia-data.nc",
    help="The input netCDF4 file with the Gaia DR3 SSO data.",
)
data_args.add_argument(
    "-o",
    "--output-name",
    type=str,
    default="./gaia-pca.nc",
    help="Output file name and path for the netCDF4 file.",
)

args = parser.parse_args()


def fill_nans(x, y, mask=None):
    """Fill in NaN values in y using cubic spline interpolation over x.

    Parameters:
    -----------
    x : 1D array-like
        The x-data
    y : 2D array-like
        The y-data with NaNs to be filled in. Each row is treated independently.
    mask : 2D boolean array-like, optional
        A mask indicating valid (True) and invalid (False) data points in y.
        If None, NaNs in y are used to create the mask.

    Returns:
    --------
    y_filled : 2D array-like
        The y-data with NaNs filled in.
    """

    if mask is None:
        mask = ~np.isnan(y)

    y = y.copy()  # to avoid modifying the original array
    for i in tqdm(range(y.shape[0]), ascii=True, desc="Filling in NaNs"):
        if ~np.isnan(y[i]).any():
            continue
        non_nan_mask = mask[i]
        cs = CubicSpline(x[non_nan_mask], y[i][non_nan_mask], bc_type="natural")
        y[i] = cs(x)

    return y


def main():
    # Load the gaia data:
    gaia_data = xr.load_dataset(args.input_name)

    # Extract relevant data
    all_gaia_nbrs = gaia_data["number"].to_numpy()
    wl = gaia_data["wl"].to_numpy()[args.wavelength_range[0] : args.wavelength_range[1]]
    refl = gaia_data["refl"].to_numpy()[
        :, args.wavelength_range[0] : args.wavelength_range[1]
    ]
    mask = (
        gaia_data["mask"]
        .to_numpy()
        .astype(bool)[:, args.wavelength_range[0] : args.wavelength_range[1]]
    )
    refl = np.where(~mask, refl, np.nan)

    # Filling in masked
    interp_refl = fill_nans(wl, refl, mask=~mask)

    # PCA setup
    all_pca_vals = np.zeros((len(args.snr_thresholds), args.num_pcs, len(all_gaia_nbrs)))

    # Perform PCA for each SNR threshold
    for i, snr_t in enumerate(args.snr_thresholds):
        snr_mask = gaia_data["snr"] > snr_t  # create a mask to have only snr > threshold
        pca = PCA(n_components=args.num_pcs)
        pca = pca.fit(interp_refl[snr_mask])  # fit PCA on the selected data
        all_pca_vals[i] = pca.transform(
            interp_refl
        ).T  # calculate PCA values for all data

    # Construct xarray dataset to store the pca values:
    all_pca_vals = xr.DataArray(
        da.from_array(all_pca_vals, chunks=(1, args.num_pcs, 1000)).astype(np.float32),
        dims=["snr_thresh", "pc", "number"],
        coords={
            "snr_thresh": args.snr_thresholds,
            "pc": np.arange(all_pca_vals.shape[1]),
            "number": all_gaia_nbrs,
        },
        name="pca_vals",
        attrs={
            "description": "PCA values for the spectra",
            "long_description": "The PCA is fitted to spectra with SNR > snr_thresh only and transformed to all spectra.",
        },
    )

    interp_refl = xr.DataArray(
        da.from_array(interp_refl, chunks=(1000, len(wl))),
        dims=["number", "wl"],
        coords={"number": all_gaia_nbrs, "wl": wl},
        name="refl",
        attrs={
            "description": "Interpolated reflectance values for the spectra",
            "long_description": "The nans in the reflectance values are filled using a cubic spline interplolation fitted to each spectrum.",
        },
    )

    # Add to dataset and save
    gaia_data["pca_vals"] = all_pca_vals
    gaia_data["interp_refl"] = interp_refl
    gaia_data["number"].attrs["description"] = "Asteroid number."
    gaia_data["wl"].attrs["description"] = "Wavelengths in µm."
    gaia_data["snr_thresh"].attrs["description"] = "SNR thresholds used for PCA fitting."
    gaia_data["pc"].attrs["description"] = "Principal component index."

    gaia_data.to_netcdf(args.output_name)   # save to netCDF4 file


if __name__ == "__main__":
    main()

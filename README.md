# Gaia PCA
The principal components of Gaia DR3 SSO spectra are known to encode the spectral type of asteroids ([Delbo et al. 2025](https://arxiv.org/abs/2511.00902)). This repository contains the code to compute the principal components of Gaia DR3 SSO spectra.

The `gaia-data.nc` file contains the Gaia DR3 SSO spectra, along with the SNR, spectral slope, and z-i color for each spectrum.

The `pca.py` script computes the principal components of the spectra and saves them to a new NetCDF file. Run the script with `-h` to see the default and available options.

## Attributions

- [valid-31oct21-mod.dat](./valid-31oct21-mod.dat): Data from [Gaia Collaboration 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...674A..35G/abstract)
- [slopeDeMeo.dat](./slopeDeMeo.dat): Provided by Marco Delbo.


## Requirements
- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- Xarray
- Dask
- TQDM


import numpy as np
import h5py
from astropy import wcs, units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt

# === Step 1: Read the HDF5 file and extract the data ===

# File path (adjust as needed)
hdf5_file = '/Users/wasi/Downloads/galaxy_galplane_l50_b0.hdf'

# Open the file and read the relevant datasets.
with h5py.File(hdf5_file, 'r') as f:
    # Read the world coordinates (assumed to be stored as 'RA' and 'DEC')
    ra  = f['RA'][:]    # in degrees
    dec = f['DEC'][:]   # in degrees

    # Read CASTOR data in the UV and g bands.
    # (The dataset names are assumed to be as in the documentation.)
    castor_uv_mag = f['CASTOR_uv_app'][:]
    castor_g_mag  = f['CASTOR_g_app'][:]

# For the purpose of making an image, we’ll work with one band.
# (You could repeat the process for the UV band or even create a multi–extension FITS file.)
# Here we use the g band.
mag = castor_g_mag

# === Step 2: Convert magnitudes to fluxes ===
# You need to adopt a zeropoint for your synthetic photometry. Here we assume a simple conversion.
# (Flux here is an arbitrary “counts” scale; adjust the zeropoint as appropriate.)
zp = 25.0  # example zeropoint
flux = 10**(-0.4 * (mag - zp))

# === Step 3: Define a WCS and create an empty image grid ===

# Determine the RA and DEC extent from the catalogue.
ra_min, ra_max = np.min(ra), np.max(ra)
dec_min, dec_max = np.min(dec), np.max(dec)

# Define a pixel scale.
# For example, 1 arcsec per pixel (~1/3600 degree).
pixel_scale = 1.0 / 3600.0  # in degrees per pixel

# Determine image dimensions (add a small margin)
naxis1 = int((ra_max - ra_min) / pixel_scale) + 10
naxis2 = int((dec_max - dec_min) / pixel_scale) + 10

# Create a WCS object.
w = wcs.WCS(naxis=2)
w.wcs.crval = [ra_min, dec_min]         # reference coordinate (e.g. lower left corner)
w.wcs.crpix = [1, 1]                    # reference pixel (1-indexed for FITS)
w.wcs.cdelt = [pixel_scale, pixel_scale]  # degrees per pixel
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

# Create an empty image array.
image = np.zeros((naxis2, naxis1))

# === Step 4: Project the catalogue onto the image grid ===

# Convert RA, DEC positions to pixel coordinates.
# Note: the 'origin' argument here is 1 to be consistent with the WCS definition.
pix_coords = w.wcs_world2pix(np.column_stack([ra, dec]), 1)
x_pix = pix_coords[:, 0].astype(int)
y_pix = pix_coords[:, 1].astype(int)

# Deposit each star’s flux into the corresponding pixel.
# (If multiple stars fall in the same pixel, their fluxes will be summed.)
for xi, yi, f_val in zip(x_pix, y_pix, flux):
    if 0 <= xi < naxis1 and 0 <= yi < naxis2:
        image[yi, xi] += f_val

# === (Optional) Convolve with a PSF kernel ===
# Convolve with a Gaussian kernel to simulate a realistic point-spread function.
psf_sigma = 0.15  # in pixels; adjust as needed
kernel = Gaussian2DKernel(x_stddev=psf_sigma)
image_conv = convolve(image, kernel)

# For the star-finding step, we will use the convolved image.
final_image = image_conv

# === Step 5: Save the projected image to a FITS file ===

# Create a FITS PrimaryHDU with the image data and WCS header.
hdu = fits.PrimaryHDU(data=final_image, header=w.to_header())
fits_filename = 'projected_g_band.fits'
hdu.writeto(fits_filename, overwrite=True)
print("Saved projected image to:", fits_filename)

# === Step 6: Run DAOStarFinder to detect stars ===

# Estimate background statistics.
mean_val, median_val, std_val = sigma_clipped_stats(final_image, sigma=3.0)

# Initialize DAOStarFinder.
# fwhm: approximate full–width at half–maximum of stars in pixels.
# threshold: detection threshold in sigma above the background.
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std_val)

# Run the star finder on the background–subtracted image.
sources = daofind(final_image - median_val)

# Print out the detected sources.
print("Detected sources:")
print(sources)

if sources is not None:
    from astropy.visualization import simple_norm
    norm = simple_norm(final_image, 'sqrt', percent=99.5)
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image, origin='lower', cmap='gray', norm=norm)
    plt.scatter(sources['xcentroid'], sources['ycentroid'],
                s=30, edgecolor='red', facecolor='none', label='DAOStarFinder')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.title('Detected Stars')
    plt.legend()
    plt.show()
else:
    print("No stars detected.")

import pyxel
config = pyxel.load("../config/g_band.yaml")

exposure = config.exposure
detector = config.detector
pipeline = config.pipeline

result = pyxel.run_mode(
    mode=exposure,
    detector=detector,
    pipeline=pipeline,
)
pyxel.display_detector(detector)


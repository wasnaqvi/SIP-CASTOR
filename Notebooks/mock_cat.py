import numpy as np
import h5py
from astropy import wcs, units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.detection import DAOStarFinder
from astropy.nddata import Cutout2D
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
zp = 2 
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

print("Image stats before convolution:")
print("  Sum:", np.sum(image))
print("  Max:", np.max(image))
print("  Min:", np.min(image))
print("  Nonzero count:", np.count_nonzero(image))


# === (Optional) Convolve with a PSF kernel ===
# Convolve with a Gaussian kernel to simulate a realistic point-spread function.
psf_sigma = 0.5  # in pixels; adjust as needed
kernel = Gaussian2DKernel(x_stddev=psf_sigma)
image_conv = convolve(image, kernel)

# For the star-finding step, we will use the convolved image.
final_image = image_conv

print("Final image stats after convolution:")
print("  Sum:", np.sum(final_image))
print("  Max:", np.max(final_image))
print("  Min:", np.min(final_image))
print("  Nonzero count:", np.count_nonzero(final_image))

# === Step 5: Save the projected image to a FITS file ===

# Create a FITS PrimaryHDU with the image data and WCS header.
hdu = fits.PrimaryHDU(data=final_image, header=w.to_header())
fits_filename = 'projected_g_band.fits'
# hdu.writeto(fits_filename, overwrite=True)
# print("Saved projected image to:", fits_filename)


# === Step 6: Run DAOStarFinder to detect stars ===

# Estimate background statistics.
mean_val, median_val, std_val = sigma_clipped_stats(final_image, sigma=3.0)

# Initialize DAOStarFinder.
# fwhm: approximate full–width at half–maximum of stars in pixels.
# threshold: detection threshold in sigma above the background.
daofind = DAOStarFinder(fwhm=2.0, threshold=2*std_val)

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
                s=30, edgecolor='red', facecolor='none', label='Brightest Stars >> 4 AB mag')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.xlim(0, 6000)
    plt.title('Detected Stars')
    plt.legend()
    plt.show()
    
    # Get a simple cutout of a section of the image around the detected stars. X-0:2000, and Y-2000:4000
    position = (1000, 3000)
    size = (2000, 2000)
    simple_cutout = Cutout2D(final_image, position=position, size=size, wcs=w)
    plt.imshow(simple_cutout.data, origin='lower', cmap='gray', norm=norm)
    # view shape of simple_cutout.data
    print(simple_cutout.data.shape)
    # Mark the detected stars on the cutout
    plt.scatter(sources['xcentroid'] - position[0], sources['ycentroid'] - position[1],
                s=30, edgecolor='red', facecolor='none', label='<4 AB mag')
    # set xlim and ylim
    plt.xlim(0,2000)
    plt.ylim(0,2000)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.title('Faint Milky Way Stars in the Galactic Plane(g band)')
    plt.show()
    # save the image to a fits file
    hdu=fits.PrimaryHDU(data=simple_cutout.data, header=simple_cutout.wcs.to_header())
    hdul=fits.HDUList([hdu])
    hdul.writeto('simple_cut.fits', overwrite=True)
    print("Saved simple cutout image to:", 'simple_cut.fits')
     # Get the pixel coordinates from DAOStarFinder
    x_coords = sources['xcentroid']
    y_coords = sources['ycentroid']
    
    # Compute the bounding box (min and max pixel positions)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Add a margin (in pixels) to the bounding box for a nicer cutout.
    margin = 10  # adjust as needed
    # Define the center and size of the cutout
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    size_x = (x_max - x_min) + 2 * margin
    size_y = (y_max - y_min) + 2 * margin
    
    position = (center_x, center_y)
    size = (size_x, size_y)
    
 
    cutout = Cutout2D(final_image, position=position, size=size, wcs=w)
    
    # --- Create a new image for only the detected stars ---
    # Instead of using the full final_image, we make a new array with the size of the cutout.
    stars_image = np.zeros_like(cutout.data)
    
    # The cutout object gives us the bounding box of the cutout in the original image.
    # Get the bounding box in the original image coordinates
    (ymin, ymax), (xmin, xmax) = cutout.bbox_original

    # Create slice objects for the original image region corresponding to the cutout:
    y_slice = slice(ymin, ymax)
    x_slice = slice(xmin, xmax)
    # Deposit each star’s flux into the new cutout image at the relative position.
    for star in sources:
        # Use the DAOStarFinder centroids
        x_star = star['xcentroid']
        y_star = star['ycentroid']
        
        # Check if the star falls within the cutout region in the original image.
        if (x_star >= x_slice.start and x_star < x_slice.stop and
            y_star >= y_slice.start and y_star < y_slice.stop):
            # Convert to the cutout image coordinate system.
            new_x = int(np.round(x_star - x_slice.start))
            new_y = int(np.round(y_star - y_slice.start))
            stars_image[new_y, new_x] += star['flux']
    
    # --- Save the stars–only cutout image to a FITS file ---
    hdu = fits.PrimaryHDU(data=stars_image, header=cutout.wcs.to_header())
    fits_filename = 'stars_cutout.fits'
    hdu.writeto(fits_filename, overwrite=True)
    print("Saved stars cutout image to:", fits_filename)
    
    # --- Display the cutout image ---
    # plt.figure(figsize=(8, 8))
    # norm = simple_norm(stars_image, 'sqrt', percent=99.5)
    # plt.imshow(stars_image, origin='lower', cmap='gray', norm=norm)
    # plt.colorbar(label='Flux')
    # plt.title('Cutout Image of Detected Stars')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()
    
    
#     hdu_stars = fits.PrimaryHDU(data=stars_image, header=w.to_header())
#     fits_filename_stars = 'stars_image.fits'
#     hdu_stars.writeto(fits_filename_stars, overwrite=True)
#     print("Saved stars image to:", fits_filename_stars)
#     plt.figure(figsize=(8, 8))
#     norm = simple_norm(stars_image, 'sqrt', percent=99.5)
#     plt.imshow(stars_image, origin='lower', cmap='viridis', norm=norm)
#     plt.colorbar(label='Flux')
#     plt.title('Image of Detected Stars')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')
#     plt.show()

# # view and open simple_cutout.fits


# import pyxel
# config = pyxel.load("../config/g_band.yaml")

# exposure = config.exposure
# detector = config.detector
# pipeline = config.pipeline

# result = pyxel.run_mode(
#     mode=exposure,
#     detector=detector,
#     pipeline=pipeline,
# )
# pyxel.display_detector(detector)

# vals=result['photon'].to_numpy()

# # investigate vals
# print(vals)
# print(vals.shape)
# print(vals[0])
# print(vals[1])
# print(vals[2])
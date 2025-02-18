from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)

# Open the FITS file
filename = 'projected_g_band.fits'
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# Use a ZScale normalization (similar to DS9's zscale)
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch='sqrt')

plt.figure(figsize=(10, 8))
plt.imshow(data, origin='lower', cmap='gray', norm=norm)
plt.colorbar(label='Flux')
plt.title("Projected g-band Image with ZScale (sqrt stretch)")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()

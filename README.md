# Simulation In-orbit Performance for CASTOR

This repository contains the **Detector Simulation Pipeline** for the *Cosmological Advanced Survey Telescope for Optical and UV Research (CASTOR)*. CASTOR is a proposed Canadian Space Agency (CSA) mission that aims to image the skies at ultraviolet (UV) and blue-optical wavelengths. This pipeline is specifically designed to simulate the in-orbit performance of the CASTOR telescope's detectors.

## About CASTOR

The CASTOR mission will provide high-resolution, wide-field imaging at ultraviolet and blue-optical wavelengths. It is expected to bridge the gap between UV imaging and other space telescopes, complementing existing and future space missions such as the Hubble Space Telescope (HST) and the James Webb Space Telescope (JWST).

For more information on CASTOR, visit the official website: [CASTOR Mission Website](https://www.castormission.org)

## Features of the Pipeline

The pipeline simulates various aspects of the CASTOR telescope’s in-orbit performance, including:

- **UV and Blue-Optical Wavelength Imaging**: Simulates the detection of astronomical objects in UV and blue-optical wavelengths.
- **Noise Models**: Incorporates realistic noise models including thermal noise, cosmic ray impacts, and readout noise.
- **Throughput Calculations**: Simulates the throughput of the telescope and detectors, accounting for the instrument's optics and detector quantum efficiency.
- **Point Spread Function (PSF)**: Models the PSF of the telescope, including aberrations and diffraction effects.
- **Orbital Effects**: Simulates the impact of orbital conditions on the detector’s performance.
  **STILL IN DEVELOPMENT**

## Requirements

To run the detector simulation pipeline, you will need the following:

- **Python 3.5** or higher
- **Pyxel 2.0.0** or higher
- **NumPy** and **SciPy** for numerical calculations
- **Matplotlib** and **bokeh** for visualizations
- **Astropy** for handling astronomical data
- **poppy**

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/wasnaqvi/SIP-CASTOR/
    ```

2. Navigate to the project directory:

    ```bash
    cd SIP-CASTOR/data
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
   **I am working on the exact requirements right now.**

## Usage

To run the notebooks:

1. Prepare the input parameters, such as the detector’s configuration, exposure time, and the model in question.
   
2. Run the jupyter notebooks.
   
3. The output will be a simulated image with the in-orbit conditions modeled, including noise and PSF effects.

   The Simulated Survey Files are available in hdf format on the Canadian Astronomical Data Center. Email me @ wasi.naqvi@nrc-cnrc.gc.ca or wasi14@student.ubc.ca for these files if you need to run Phase_0 sim.ipynb

## Configuration

The configuration file `xx.yaml` contains the parameters for the simulation. The .yaml files follow Pyxel's JSON schema.



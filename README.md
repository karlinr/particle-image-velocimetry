# Particle Image Velocimetry

A Python implementation of particle image velocimetry (PIV) developed as part of my master's project. My work was on quantifying zebrafish blood flow velocity and quantifying the uncertainty on those measurements. Full details at [https://github.com/karlinr/particle-image-velocimetry/blob/75ed359d2e0533602451887c8543f1f8d1c84639/](https://github.com/karlinr/particle-image-velocimetry/blob/75ed359d2e0533602451887c8543f1f8d1c84639/)

## Usage

### Basic PIV analysis

```python
from classes.piv import PIV

# Create a PIV object
# Parameters: title, interrogation window size, search area size, vector spacing, intensity threshold, peak-finding method
piv = PIV("my_experiment", _iw=24, _sa=24, _inc=16, _threshold=0.4, _pfmethod="5pointgaussian", _pad=True)

# Load a TIFF video (frames alternate: frame A, frame B, frame A, frame B, ...)
piv.add_video("data/my_video.tif")

# Set up a regular grid of interrogation windows
piv.get_spaced_coordinates()

# Compute correlation matrices for all window locations and frame pairs
piv.get_correlation_matrices()

# Average correlations across frame pairs and find the velocity field
piv.get_correlation_averaged_velocity_field()

# Plot the resulting flow field
piv.plot_flow_field()
```

### Single-location analysis

```python
# Analyse a single interrogation window at pixel coordinate (x, y)
piv.set_coordinate(196, 234)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
```

### Bootstrap uncertainty estimation

```python
# Estimate velocity uncertainty by resampling frame pairs
std_x, std_y = piv.get_uncertainty(n_bootstrap=1000)
```

### Multi-pass with window deformation

```python
# First pass
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.get_velocity_field()

# Apply window deformation and repeat with a smaller search area
piv.do_pass()
piv.set(_iw=24, _sa=8, _inc=1)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The zebrafish data consists of light-sheet fluorescence microscopy TIFF stacks acquired over multiple cardiac cycles. Each file contains alternating frame pairs used for PIV. Raw data is pre-processed with `data/process_zebra.py` to extract phase-matched frame pairs for correlation averaging.

Synthetic test data can be generated with `data/create_simulation.py` for validation and algorithm development.

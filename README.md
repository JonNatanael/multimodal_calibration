# Multimodal calibration
This repository contains the reference implementation of the approach presented in the paper *Joint Calibration of a Multimodal Sensor System for Autonomous Vehicles* by Jon Muhovič and Janez Perš.

bibtex:
>@article{muhovivc2023joint,
  title={Joint Calibration of a Multimodal Sensor System for Autonomous Vehicles},
  author={Muhovi{\v{c}}, Jon and Per{\v{s}}, Janez},
  journal={Sensors},
  volume={23},
  number={12},
  pages={5676},
  year={2023},
  publisher={MDPI}
}

## What this code does
The calibration pipeline aligns a LiDAR point cloud to a camera by optimizing a 6-DOF rigid transform (translation + roll/pitch/yaw) so projected LiDAR edges match image edges of a calibration target. It:

1. Loads camera intrinsics and distortion from a YAML calibration file.
2. Loads image/LiDAR pairs, finds the asymmetric circle-grid calibration target, and derives an edge map for the target in the image.
3. Extracts LiDAR edge candidates that correspond to the target plane.
4. Iteratively optimizes the LiDAR-to-camera transform by maximizing overlap between projected LiDAR edges and the image edge map.
5. Saves the resulting extrinsics to a YAML file in `calibrations/`.

## Repository layout
- `main.py`: entry point for optimization (`calibrate_extrinsics`).
- `utils.py`: calibration utilities (data loading, edge extraction, projection, scoring).
- `config.yaml`: configuration for dataset paths, optimization settings, and initial parameters.
- `calibrations/`: output directory for LiDAR-to-camera extrinsic YAML files.

## Requirements
Install the required Python packages (Python 3.8+ recommended):

```bash
pip install numpy opencv-python matplotlib pyyaml
```

## Data expectations
The code expects a dataset structure like:

```
<repo_root>/data/<camera_name>/
  frame_0001.png (or .jpg)
  frame_0001.npy
  frame_0002.png
  frame_0002.npy
  ...
```

Each `.npy` file should be a NumPy object array containing a dictionary with a `pc` key. The `pc` value should be an `N x 4` array with XYZ coordinates in columns 0–2 and the LiDAR ring/beam index in column 3 (used for edge extraction).

Camera intrinsics must be stored in `calibrations/<camera_name>.yaml` using OpenCV `FileStorage` keys:

- `imageSize`
- `cameraMatrix`
- `distCoeffs`

## Configuration
Edit `config.yaml` to point to your data, select the active camera, and tune optimization parameters. The relevant settings include:

- `GENERAL.data_dir`: path to the dataset root (default: `data`).
- `GENERAL.calibration_dir`: where camera intrinsics live (default: `calibrations`).
- `GENERAL.camera_name`: active camera folder name (e.g., `zed`, `thermal_camera`).
- `OPTIMIZATION.*`: number of iterations, batch size, learning rate, and step sizes.
- `INITIAL_PARAMETERS.<CAMERA>.PARAMS`: initial [x, y, z, roll, pitch, yaw] estimate in meters/degrees.

## Usage
1. Prepare camera intrinsics and data for your camera name.
2. Update `config.yaml` with the desired `GENERAL.camera_name` and initial parameters.
3. Run the calibration:

```bash
python main.py
```

When finished, the optimized extrinsics are written to `calibrations/<camera_name>_lidar.yaml`.

## Tips
- Enable/disable live visualization with `DISPLAY.show_progress` in `config.yaml`.
- If you see “calibration not found!”, verify the path and keys in the camera intrinsics YAML.
- If you see “no data found”, confirm that image and `.npy` filenames align and are present under `data/<camera_name>/`.

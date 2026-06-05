# Notebooks

This directory contains Jupyter notebooks demonstrating the main
Stitcher v0.2 workflows.

| Notebook                              | Description |
|---------------------------------------|-------------|
| `01_quickstart.ipynb`                 | Minimal end-to-end example on **synthetic** data. |
| `02_full_pipeline_batch_1_EL.ipynb`   | The original DanMAX "extended layer" workflow.    |
| `03_stitching_with_rotation.ipynb`    | Stitching on a rotation-stage dataset.            |

## Running the notebooks

1. Install the package with the notebook extras:

   ```bash
   pip install -e ".[notebook,danmax]"
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `01_quickstart.ipynb` first — it is self-contained and uses
   synthetic data, so you do not need any beamline files to run it.

## Notes on `02_*` and `03_*`

These are the **original** beamline notebooks, lightly cleaned up. They
import the `stitcher` package instead of the old `volume_stitching_3d_v0_2`
module, but they still reference DanMAX paths such as
`/data/visitors/danmax/20240533/2024101108/...`. Update those paths to
point to your own data before running.

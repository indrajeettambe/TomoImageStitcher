# Notebooks

Jupyter tutorials for TomoImageStitcher.
The notebooks are the **canonical place to learn the pipeline**;
the README only gives a 10-line overview.

| Notebook | What it covers | Data |
|----------|----------------|------|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Minimal end-to-end example, 5 lines of code. | synthetic |
| [`02_full_pipeline.ipynb`](02_full_pipeline.ipynb) | **Detailed step-by-step walkthrough of every one of the six pipeline stages**, with explanations, parameter choices, and visualisations. **Start here.** | synthetic |
| [`03_stitching_with_rotation.ipynb`](03_stitching_with_rotation.ipynb) | Original DanMAX rotation-stage workflow. | beamline paths (edit before running) |

## Running the notebooks

1. Install the package with the notebook extras:

   ```bash
   pip install -e ".[notebook,danmax]"
   pip install cupy-cuda12x        # or cupy-cuda11x to match your CUDA
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `02_full_pipeline.ipynb` first. It is self-contained, uses
   synthetic data, and runs end-to-end on a laptop GPU in under a minute.

4. When you are ready to run on real data, edit the top of the notebook
   to point at your own `.h5` files and motor coordinates.

## Notes on `03_stitching_with_rotation.ipynb`

This is the **original beamline notebook** from the DanMAX experiment.
It is kept as a reference for the rotation-stage workflow. Before
running it, edit the file paths in cell 1 to point at your own data
(the notebook currently references the DanMAX directory layout).

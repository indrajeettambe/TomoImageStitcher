# Experimental bin4 open-cell foam workflow

This dataset-specific CPU workflow documents the registration and edge-aware reassembly used for a five-step open-cell foam dataset. It is **experimental** and is not a general replacement for `Stitcher`. Low-contrast overlaps remain uncertain, and neither the workflow nor its sampled checks establish that every ligament or precipitate is preserved. It performs no denoising, thresholding, or intensity equalization. It retains the original bin4 intensity scale, but interpolation and overlap blending can alter small features. Keep all source reconstructions and previous outputs.

The workflow was developed using 38 tiles per layer, four layers per step, 304 × 304 × 324 float32 input tiles at 4.4 µm per voxel, and datasets named `0000um`, `0250um`, `0500um`, `0750um`, and `1000um`. The circular input support was observed to be approximately 151 pixels in radius; that mask must be re-evaluated for any other input. No scan data, registration records, or results are included in this repository.

## Inputs and layout

Copy the contents of `workspace/` into a **new independent workspace** and place the experiment's pre-existing saved files under each step:

```
workspace/
  0000um/ ... 1000um/
    layers/layer0.h5 ... layer3.h5
    registration/layer0.npz ... layer3.npz
    registration/layer0_neighbours.json ... layer3_neighbours.json
  review_20261103/          # scripts and generated registration JSON
  candidate_restitch/code/ # assembly and validation scripts
```

The saved layer files must contain `/stitched_data/stitched_image` and `/stitched_data/layer_final_global_coordinates`. Each layer registration NPZ must provide `coords_px`, `paths`, and other metadata as used by the scripts; tile input HDF5 files must contain `/exchange/data`. Existing `paths` entries must point to readable reconstruction files. Coordinate systems and the ordering of axes must be verified independently; in the original experiment the first slices of a numbered layer matched the last slices of the next layer. Using that ordering blindly on another dataset can destroy continuity.

## Steps

Run from the workspace root with Python, NumPy, SciPy, scikit-image, h5py and Matplotlib available:

1. `python review_20261103/measure_vertical.py` measures possible Z offsets on saved mosaic overlaps.
2. For each step and adjacent layer pair, run `python review_20261103/measure_raw_tiles.py --step 0000um --pair 0` with pair values 0–2. This uses recorded bin4 input paths and writes tile-match diagnostics.
3. For each step and layer, run `python review_20261103/horizontal_overlap_audit.py --step 0000um --layer 0`, with layer values 0–3.
4. For each step, run `python review_20261103/solve_positions.py 0000um` and `python review_20261103/evaluate_position_independent.py 0000um`. Repeat the horizontal audit with `--solution` to assess whether the corrections worsen previously good overlaps. Confidence thresholds are dataset-specific, and the graph is regularized toward prior positions where measurements are missing.
5. Only after reviewing registrations, run `python candidate_restitch/code/reconstruct_edge.py 0000um`. The assembler writes a `.partial` file, records geometric coverage and provenance, and renames the file only when it has written the complete volume. It refuses to overwrite a completed output. An interrupted HDF5 file may be corrupt and should not be assumed resumable.
6. Run `python candidate_restitch/code/verify_edge_volume.py 0000um`. This recomputes a few output crops from the source tiles and checks numerical equality. It is an implementation check, not a scientific validation of alignment or feature preservation. `test_edge_interpolation.py` checks constant-signal behavior at a partially covered boundary. The coverage and plotting scripts can help locate seams, but their thresholds and display limits must be inspected before use.

Repeat the steps for the other four step names. The outputs are provisional until representative weak ligaments, precipitates, low-confidence tile pairs, gaps, and seams have been compared with the source data. Do not delete earlier results based only on a successful run.

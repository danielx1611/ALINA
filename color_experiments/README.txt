ALINA Assignment 5: ROI-masked end-to-end evaluation

Replace color_experiments/run_end_to_end.py with the enclosed file.
Do not modify the trained models or existing annotations.

From the ALINA repository root, run:

python color_experiments/run_end_to_end.py --frames-per-video 5 --methods baseline svm --output outputs/color_experiments/end_to_end_roi_fixed

The script now removes predictions outside the selected and perspective-warped ROI.
It also raises the connected candidate limit from 10,000 to 100,000 pixels.
Results include mask coverage across the whole frame and within the valid ROI.
Skipped excessive components remain explicitly flagged, not counted as detections.

The new output directory is required because existing results were generated
with different settings and should not be mixed with the new experiment.

The script's syntax has been checked. Full testing requires your local ALINA
frames, ROI coordinates, and saved models.

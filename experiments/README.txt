ALINA automatic ROI restart

Replace experiments/roi_methods.py and experiments/run_experiments.py with these files.
Run from ALINA repository root:
  python experiments/run_experiments.py --methods hsv hough ransac kmeans
  python experiments/run_experiments.py --methods gemini

Selection images are resized using ALINA's own compute_display_scale(max_display_dim=1080).
ROI points are scaled back to original-resolution coordinates before process_image.
Gemini receives the resized image and its exact dimensions. No ROI-area cap or
90,000-pixel connected-component safety check is applied. Invalid coordinates
are rejected rather than silently clipped. ALINA's original processing remains
unchanged and may run for a long time with poor ROI selections.

New output directory: outputs/automated_experiments_v2
Gemini cache includes selection dimensions and model name; old cached full-res
predictions will not be reused. Do not commit API keys.

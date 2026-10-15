ALINA Single-Image ROI Tests

This folder contains earlier tests made while developing the automatic ROI methods. It tests one image at a time. The main assignment results were collected later with dynamic_experiments/run_dynamic_batch.py.

Run these commands from the ALINA project root:
  python experiments/run_experiments.py --methods hsv hough ransac kmeans
  python experiments/run_experiments.py --methods gemini

The default image is data/Raw_Data/vidd_1/00001.jpg. For another image:
  python experiments/run_experiments.py --image data/Raw_Data/vidd_2/00001.jpg --methods hough

Gemini needs GEMINI_API_KEY in the environment. The default model is gemini-flash-lite-latest, and another available model can be set with --gemini-model.

The input image is resized for ROI selection (maximum dimension 1080 by default). The coordinates are converted back to the original resolution before ALINA runs. Results go to outputs/automated_experiments_v2/ unless you set --output-dir.

These early previews are useful for checking individual ROI predictions and ALINA results, but use dynamic_experiments/ for the main frame-by-frame results.

ALINA Dynamic ROI Experiments

This folder contains the main Assignment 4 experiments. Each frame gets its own automatically selected trapezoid, which is passed to the original ALINA pipeline. ALINA's line detection code is unchanged.

The methods are HSV, Hough, RANSAC, K-Means, and Gemini. The first four methods process all JPG frames in data/Raw_Data/vidd_1, vidd_2, and vidd_3. Gemini processes 10 evenly spaced frames per video by default.

Run everything from the ALINA project root with your virtual environment active.

Install the extra libraries if needed:
  python -m pip install -e .
  python -m pip install scikit-learn google-genai pandas

Run the traditional methods:
  python dynamic_experiments/run_dynamic_batch.py --methods hsv hough ransac kmeans

For Gemini, first set GEMINI_API_KEY in your terminal. In PowerShell:
  $env:GEMINI_API_KEY = "YOUR_KEY_HERE"
  python dynamic_experiments/run_dynamic_batch.py --methods gemini

The default Gemini model is gemini-flash-lite-latest. Use --gemini-model MODEL_NAME to choose another available model. If you change the model or other settings, use --output-dir outputs/another_run so results from different experiments are not mixed.

Useful options:
  --data-dir              Input parent folder (default: data/Raw_Data)
  --output-dir            Where results are saved (default: outputs/dynamic_experiments)
  --gemini-per-video      Number of Gemini frames from each video (default: 10)
  --seed                  Random seed (default: 42)
  --max-selection-dim     Maximum size used for selecting ROIs (default: 1080)
  --retry-failed          Retry previously failed evaluations

The script writes results.csv and experiment_config.json in the output folder. Each frame/method folder also contains the ROI coordinates, an ROI preview, ALINA's annotated image, and detected pixel coordinates. The CSV is updated as each run finishes, so interrupted batches can be continued.

A frame with detected pixels is not proof of pixel-level accuracy. Some valid-looking ROIs still produce no ALINA annotations. Processing can also take a long time on certain ROIs.

Make the graphs with:
  python dynamic_experiment_analysis/analyze_dynamic.py

See dynamic_experiment_analysis/README.txt for more details.

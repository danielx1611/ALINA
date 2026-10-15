ALINA DYNAMIC ROI BATCH EXPERIMENTS

Copy run_dynamic_batch.py and roi_methods.py into ALINA/experiments/.
Do not modify alina/pipeline.py or cli.py.
Run these commands from the ALINA repository root with the virtual environment active:

python experiments/run_dynamic_batch.py --methods hsv hough ransac kmeans
python experiments/run_dynamic_batch.py --methods gemini

Default: all JPG frames in data/Raw_Data/vidd_* for traditional methods;
10 evenly spaced frames from each video for Gemini.

Gemini requires GEMINI_API_KEY (unless predictions are already cached).
Default Gemini model matches the supplied single-frame runner: gemini-flash-lite-latest.
To use gemini-3-flash-preview instead:
python experiments/run_dynamic_batch.py --methods gemini --gemini-model gemini-3-flash-preview --output-dir outputs/dynamic_gemini_3

The script saves outputs/dynamic_experiments/results.csv after EVERY frame/method.
Rerunning skips completed rows, including failures by default. To retry failures:
python experiments/run_dynamic_batch.py --retry-failed
The raw Gemini API response is cached per video/frame/model/resolution, so retrying a
failed ALINA step does not require another API call. A separate inference-time JSON
is saved for fresh Gemini requests; cached ROI time is not actual model latency.

CAUTION: Original ALINA CIRCLEDAT may run very slowly on certain ROIs.
Use Ctrl+C to interrupt if needed. Previously completed frames remain recorded.

Detected pixel count and successful annotation are NOT accuracy metrics.
For precision, recall, and F1, evaluate against manually annotated CBEM ground truth.
Fixed-ROI experiments are intentionally excluded, per request.

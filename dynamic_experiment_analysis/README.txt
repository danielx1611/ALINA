Place analyze_dynamic.py in ALINA/analysis/ and run from the ALINA repository root:

  python -m pip install pandas matplotlib
  python analysis/analyze_dynamic.py

Outputs in outputs/dynamic_analysis/:
- overall_summary.csv
- processing_time_summary.csv
- per_video_summary.csv
- figure_1_detection_success.png
- figure_2_processing_time.png
- figure_3_per_video.png
- cbem_per_frame.csv
- cbem_summary.csv
- cbem_missing_predictions.csv

Gemini cached ROI timings are excluded from the timing chart. When only a few
live API timings exist, note the sample count in processing_time_summary.csv.

CBEM evaluation uses exact (x,y) matching, not the ALINA repository's
separate-X/Y averaging method. Check image resolution and source alignment.
Missing ground-truth frames and missing Gemini predictions are NOT counted as zeros.

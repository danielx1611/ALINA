ALINA Dynamic Experiment Analysis

This folder has the script used to make the summary tables and graphs for Assignment 4. It reads the completed experiment CSV and does not rerun ALINA or the ROI selection methods.

Run these commands from the ALINA project root:
  python -m pip install pandas matplotlib
  python dynamic_experiment_analysis/analyze_dynamic.py

By default, it reads outputs/dynamic_experiments/results.csv and saves everything under outputs/dynamic_analysis/.

Main outputs:
  overall_summary.csv                Detection totals and success rates
  processing_time_summary.csv        ROI selection and ALINA processing time
  per_video_summary.csv              Detection success in each video
  figure_1_detection_success.png     Detection success by method
  figure_2_processing_time.png       Mean processing time
  figure_3_per_video.png             Detection success by video

Gemini's processing-time average may use fewer measured samples than the traditional methods. Check processing_time_summary.csv for the sample counts.

The script also produces optional CBEM files. These were not used for the report's accuracy claims because the provided ground-truth data did not reliably align with the experimental frames. The report focuses on the three figures above and the saved visual examples.

To analyze another folder:
  python dynamic_experiment_analysis/analyze_dynamic.py --results outputs/another_run/results.csv --experiment-root outputs/another_run --output outputs/another_analysis

The generated output files stay on your computer and are ignored by Git.

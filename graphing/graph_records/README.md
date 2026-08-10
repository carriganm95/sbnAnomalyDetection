# Graph Records

This directory records the main plots, experiments, and hyperparameter-tuning insights from the Graph-VAE anomaly-detection study. Each subdirectory focuses on a different stage of the investigation and includes its own README with more detailed observations.

## Experiment summaries

### [Dataset features](dataset_features/)

This directory contains plots describing the training dataset (`events_train`) and the good and bad testing datasets (`good_events_test` and `bad_events_test`).

- The box-and-whisker plots show the number of events in 1800-second windows with a 15-second stride. The runs form two groups with noticeably different event counts, likely because of different detector trigger settings.
- The per-channel nonzero-event histogram shows where detector activity is concentrated. Most nonzero events occur near the centers of the induction planes, while pulses are less common in the collection planes.
- This channel-occupancy pattern closely resembles the per-channel average anomaly-score pattern, suggesting that event frequency may influence the spatial structure learned by the model.

### [Good-run and bad-run histograms](good_run_bad_run_histograms/)

Initial comparisons of pulse-integral distributions from selected good and bad runs, plotted over 500-channel intervals. These plots were used for early data exploration; the selected runs are not intended to represent every possible detector condition, and differences in trigger settings may affect their event counts and pulse-integral densities.

### [Window size and stride effects](window_size_and_stride_effect/)

Studies of how event-based window size and stride affect the window-level anomaly-score distributions.

- Larger windows generally improve the separation between good and bad runs.
- Smaller strides produce smoother, more stable score distributions by increasing the overlap between consecutive windows.

### [Whole-channel studies](whole_channels/)

Per-channel and window-level results across the full detector channel range, including comparisons between 9-feature and 14-feature models.

- The general window-size trend remains: larger windows often improve good/bad separation.
- Some window sizes fall within a transition region where the per-channel score pattern changes and good/bad scores temporarily overlap.
- Expanding the input from 9 to 14 features improves separation in the collection planes. The added features are `width_mean`, `width_stdev`, `sumadc_mean`, `mult_mean`, and `sp_fraction`.

### [Timed-window studies](timed_window/)

Experiments replacing event-count windows with time-based windows to make the model easier to interpret and less dependent on the detector trigger rate.

- Larger time windows generally improve good/bad separation, while smaller time strides produce smoother and more stable distributions.
- Within an intermediate window-size range, the per-channel anomaly-score pattern changes: separation weakens and then reappears, with peaks shifted from the centers of the induction planes toward the plane junctions.
- Very long windows may encourage overfitting; the experiments identify a time window around 2000 seconds as a practical candidate.
- The training-data baseline is shown in green on the magnified per-channel plots, and the shaded bands represent ±1 standard deviation.

## Overall conclusions

Across both event-based and time-based experiments, window size and stride strongly affect Graph-VAE performance. Larger windows provide more information and usually improve good/bad separation, while smaller strides make anomaly-score distributions more stable. However, increasing the window size does not guarantee better performance: transition regions can change the spatial pattern of per-channel scores and temporarily mix the good and bad distributions.

The full-channel studies also show that richer input features are important for separating detector conditions, especially in the collection planes. Dataset-level differences in trigger rate and channel occupancy should also be considered when interpreting the model's anomaly-score patterns.

These plots are experimental records rather than a final benchmark. Results should be interpreted together with the model configuration, selected runs, channel range, feature set, and window definition used for each study.
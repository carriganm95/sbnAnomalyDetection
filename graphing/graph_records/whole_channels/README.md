# Graphe-VAE model performance with new features and different window size

The current subdirectory shows per channel average anomaly scores for both good runs and bad runs
under different model settings. We have two major findings:
- First, the findings we had in the `window_size_and_stride_effect/` still works for the whole channels.
  The larger the window size, the more separated the good and bad runs. However, for certain window
  sizes that are within the anomaly score transition range, the performance of the model is going to be low even if the window size is too large (e.g., 2000 events window size), as the model is under transition. Specifically, the per channel anomaly scores will 
  going to change in shape, and the maximum separation will start to slide from the center of induction planes to the plane junctions. (In the `timed_window/` directory, we also found this phenomenon). As a result, the good and bad runs will have very close anomaly
  scores and will be mixed together.
- Second, we found that with only 9 features (sum, min, max, mean, stdev, count, occupancy, time_mean, time_spread), it is 
  hard to separate the good run and bad run in the collection planes. Therefore, we added a few more features to reach 14 features
  (sum, min, max, mean, stdev, count, occupancy, time_mean, time_spread, width_mean, width_stdev, sumadc_mean, mult_mean, sp_fraction)
  so the model can under stand the runs and channels better. The experiment result shows that these features successfully increased the
  separation of good and bad runs in the collection plane.
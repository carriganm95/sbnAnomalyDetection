# Graphe-VAE model performance with new features and different window size

The current subdirectory shows per channel average anomaly scores for both good runs and bad runs
under different model settings. We have two major findings:
- First, the findings we had in the `window_size_and_stride_effect/` still works for the whole channels.
  The larger the window size, the more separated the good and bad runs. However, for certain window
  sizes that are within the anomaly score transition range (please check `timed_window/` directory), the 
  performance of the model is going to be low even if the window size is large, as the model is under transition.
- Second, we found that with only 9 features (sum, min, max, mean, stdev, count, occupancy, time_mean, time_spread), it is 
  hard to separate the good run and bad run in the collection planes. Therefore, we added a few more features to reach 14 features
  (sum, min, max, mean, stdev, count, occupancy, time_mean, time_spread, width_mean, width_stdev, sumadc_mean, mult_mean, sp_fraction)
  so the model can under stand the runs and channels better. The experiment result shows that these features successfully increased the
  separation of good and bad runs in the collection plane.
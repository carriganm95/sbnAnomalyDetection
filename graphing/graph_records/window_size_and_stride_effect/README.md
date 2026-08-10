# Graph-VAE model performance on different window size and stride

These are the histograms of good and bad runs (windows) under different model window size
settings and stride settings. We found two major things:
- First, when stride is fixed and window size changes, the larger the window size, the more 
  separated the good and bad runs.
- Second, when window size is fixed and stride changes, the smaller the stride the more stable
  the per window scores. (As we can see if stride is larger, the good runs are more likely
  to have higher anomaly scores than bad runs for some windows)

In our later experiment, we also have findings about the window size and per channel anomaly scores.
Please check the `whole_channels/` directory for more details.
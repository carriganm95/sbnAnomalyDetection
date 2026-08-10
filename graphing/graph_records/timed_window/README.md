# Towards a more physical model: Timed window

Before, our Graph-VAE models' window sizes are set by the number of events. However, having
window size to be number events has disadvantages. Having window size based on events is less 
straightforward than a specific time, so it hard to interpret. Also, having window size based on
events means the model will need to wait for a long time to initiate the first window when the model 
trigger is high (so fewer events in a fixed amount of time). Therefore, we decide to switch our model
to use a timed window. Now, the window size and the stride are going to be based on a specific time.
This subdirectory shows a lot of plots about our experiments of timed windows. 
Our major findings about the timed window are:
- First, our findings about window size and stride still work here. Larger window size
  makes the good and bad runs more separated, and lower stride makes the model more stable
  and the distribution of good and bad runs smoother.
- However, there is one thing to be noticed: as we slide the timed window size from small to large,
  there is a specific range (5000 seconds to 40000 seconds in our experiment) that the model starts to
  flip its per channel anomaly scores. Before, the per channel anomaly scores tend to peak around the center
  of the induction planes, but then as the timed window size gradually increases, the good and bad per channel anomaly
  scores would mix together, and then as the timed window size further increases, the good and bad per channel anomaly
  scores would reseparate. However, the new separation no longer has peaks around the center of the induction planes, but
  at the junctions of the planes. We believe such timed window size is already too large and the model already starts to
  overfit. Therefore, we believe our timed window size shouldn't be too long, and should probably be around 2000 seconds.
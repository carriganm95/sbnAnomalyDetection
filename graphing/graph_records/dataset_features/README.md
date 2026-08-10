# Plots about dataset features

This subdirectory shows certain statistics as well as features of the dataset we used, including
the events_train dataset (training), good_events_test dataset (testing), and the bad_events_test dataset (testing).
- The box whisker plots of the training and testing dataset shows the number of events in a fixed time window of
  1800 seconds (which is very close to our timed window setting of 2000 seconds in experiment) and time stride of 15 seconds.
  From these box whisker plots, we can notice that there are two kinds of runs, one with a lot more events than the other. We 
  believe such a difference represents the different detector trigger settings, causing some runs to have much more events than the
  others. 
- The per channel non zero events histogram illustartes the usual non zero event number distributions of each channels. 
  We can see the trend in the plot is very close to the per channel average anomaly scores in our experiments. From this plot we learned that most events concentrate around the center of the induction planes, while the collection planes are much rarer to have a pulse (a non zero event for a channel means there is at least a pulse in the channel during that event).
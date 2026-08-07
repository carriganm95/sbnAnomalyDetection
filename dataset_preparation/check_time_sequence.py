import numpy as np

path = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/data/events_train.npz"

with np.load(path, allow_pickle=True) as data:
    evt_time = np.asarray(data["evt_time"])
    evt_run = np.asarray(data["evt_run"])
    evt_subrun = np.asarray(data["evt_subrun"])
    evt_num = np.asarray(data["evt_num"])

bad = np.flatnonzero(np.diff(evt_time) < 0)

print(f"Total events: {len(evt_time):,}")
print(f"Number of time decreases: {len(bad):,}")

for i in bad[:20]:
    print(
        f"\nDecrease between indices {i} and {i + 1}:"
        f"\n  before: run={evt_run[i]}, subrun={evt_subrun[i]}, "
        f"event={evt_num[i]}, time={evt_time[i]}"
        f"\n  after:  run={evt_run[i + 1]}, subrun={evt_subrun[i + 1]}, "
        f"event={evt_num[i + 1]}, time={evt_time[i + 1]}"
    )
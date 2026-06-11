"""Sparse event dataset that forms windows lazily in __getitem__.

Stores all events as flat CSR-style arrays (channels_flat, integrals_flat, offsets).
This is ~1000x more compact than a pre-materialized dense window array because
most channels are inactive in any given event.

Memory estimate for 79k events with ~1k hits/event:
  channels_flat:  79M * 8 bytes =  ~632 MB
  integrals_flat: 79M * 4 bytes =  ~316 MB
  offsets:        79k * 8 bytes =  ~0.6 MB
  Total:                            ~950 MB  (vs 85 GB for dense)

Windows (history past frames + 1 target) are built on the fly using the
same vectorised np.add/minimum/maximum.reduceat logic as the materialiser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from sbn_anomaly.data.graph_window_dataset_pyg import build_sparse_edge_index

logger = logging.getLogger(__name__)

_ALLOWED_FEATURES = {"sum", "min", "max", "stdev", "mean", "count"}


def _run_subrun_evt_key(meta: dict, tpc_branches: List[str]) -> tuple:
    """Return a (run, subrun, evt) sort key from a per-event meta dict."""
    run = subrun = evtnum = None
    for b in tpc_branches:
        if b in meta:
            bl = b.lower()
            if "run" in bl and "subrun" not in bl:
                run = int(meta[b])
            elif "subrun" in bl:
                subrun = int(meta[b])
            elif "evt" in bl:
                evtnum = int(meta[b])
    return (run or 999999999, subrun or 999999999, evtnum or 999999999)


def _extract_provenance(
    raw_events: list,
    tpc_branches: List[str],
) -> tuple:
    """Return (evt_run, evt_subrun, evt_num) int32 arrays from sorted raw_events."""
    runs, subruns, evtnums = [], [], []
    for e in raw_events:
        r, s, n = _run_subrun_evt_key(e["meta"], tpc_branches)
        runs.append(r if r != 999999999 else -1)
        subruns.append(s if s != 999999999 else -1)
        evtnums.append(n if n != 999999999 else -1)
    dtype = np.int32
    if not raw_events:
        empty = np.empty(0, dtype=dtype)
        return empty, empty, empty
    return (
        np.array(runs, dtype=dtype),
        np.array(subruns, dtype=dtype),
        np.array(evtnums, dtype=dtype),
    )


class SparseWindowDatasetPyG(Dataset):
    """PyG dataset that stores raw sparse events and builds windows in __getitem__.

    Each sample covers ``(history + 1) * window_size`` consecutive events:
      - first  history * window_size  events  -> history past frames (input x)
      - last          window_size     events  -> target frame (y)

    Each frame is computed by splitting window_size events into n_bins temporal
    bins and aggregating hit integrals per channel per bin.

    Attributes
    ----------
    hit_branches : list[str]
        Node-feature names (e.g. ["sum","min","max","stdev","count"]).
        Exposed so BaseTrainer can label reconstruction plots.
    num_nodes : int
        Total channel count (including inactive channels).
    node_feat_dim : int
        Features per frame = n_bins * n_node_features.
    """

    def __init__(
        self,
        channels_flat: np.ndarray,
        integrals_flat: np.ndarray,
        offsets: np.ndarray,
        n_channels: int,
        history: int = 4,
        window_size: int = 20,
        n_bins: int = 4,
        stride: int = 1,
        radius: int = 4,
        node_features: Optional[List[str]] = None,
        prune_inactive: bool = True,
        evt_run: Optional[np.ndarray] = None,
        evt_subrun: Optional[np.ndarray] = None,
        evt_num: Optional[np.ndarray] = None,
        evt_file_idx: Optional[np.ndarray] = None,
        filenames: Optional[List[str]] = None,
        times_flat: Optional[np.ndarray] = None,
        wires_flat: Optional[np.ndarray] = None,
        planes_flat: Optional[np.ndarray] = None,
        tpcs_flat: Optional[np.ndarray] = None,
    ) -> None:
        self._channels_flat = np.asarray(channels_flat, dtype=np.int64)
        self._integrals_flat = np.asarray(integrals_flat, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._times_flat = np.asarray(times_flat, dtype=np.float32) if times_flat is not None else None
        self._wires_flat = np.asarray(wires_flat, dtype=np.int32) if wires_flat is not None else None
        self._planes_flat = np.asarray(planes_flat, dtype=np.int32) if planes_flat is not None else None
        self._tpcs_flat = np.asarray(tpcs_flat, dtype=np.int32) if tpcs_flat is not None else None

        # Optional per-event provenance (run, subrun, event number, source file)
        self._evt_run = np.asarray(evt_run, dtype=np.int32) if evt_run is not None else None
        self._evt_subrun = np.asarray(evt_subrun, dtype=np.int32) if evt_subrun is not None else None
        self._evt_num = np.asarray(evt_num, dtype=np.int32) if evt_num is not None else None
        self._evt_file_idx = np.asarray(evt_file_idx, dtype=np.int32) if evt_file_idx is not None else None
        self._filenames: List[str] = list(filenames) if filenames is not None else []

        self.num_nodes = int(n_channels)
        self.history = int(history)
        self.window_size = int(window_size)
        self.n_bins = int(n_bins)
        self.stride = int(stride)
        self.radius = int(radius)
        self.prune_inactive = bool(prune_inactive)

        if node_features is None:
            node_features = ["sum", "min", "max", "stdev", "count"]
        invalid = set(node_features) - _ALLOWED_FEATURES
        if invalid:
            raise ValueError(f"Unknown node_features: {invalid}")
        self.node_features = list(node_features)
        self.n_node_features = len(self.node_features)
        self.node_feat_dim = n_bins * self.n_node_features
        self.hit_branches = self.node_features

        n_events = len(self._offsets) - 1
        self.events_per_sample = (history + 1) * window_size
        self._starts = list(range(0, n_events - self.events_per_sample + 1, stride))

        self._bin_splits = np.array_split(np.arange(window_size), n_bins)

        logger.info(
            "Building edge index: %d nodes, radius=%d", self.num_nodes, radius
        )
        self.edge_index_full = build_sparse_edge_index(self.num_nodes, radius=radius)
        self._edge_src_np = self.edge_index_full[0].numpy().copy()
        self._edge_dst_np = self.edge_index_full[1].numpy().copy()

        self._channel_idx = (
            torch.arange(self.num_nodes, dtype=torch.float32)
            / max(1, self.num_nodes - 1)
        ).unsqueeze(1)

        # Pre-aggregate raw hits per channel per event once so __getitem__ only
        # needs to combine ~events_per_bin already-deduplicated arrays per bin
        # instead of sorting all raw hits on every sample access.
        logger.info("Pre-aggregating events (%d events)...", len(self._offsets) - 1)
        self._build_event_aggregates()

        logger.info(
            "SparseWindowDatasetPyG ready: %d samples, %d channels, "
            "%d features/frame, history=%d, window_size=%d, n_bins=%d",
            len(self._starts), self.num_nodes, self.node_feat_dim,
            self.history, self.window_size, self.n_bins,
        )

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_root(
        cls,
        root_files,
        tree_name: str,
        hit_branches: List[str],
        n_channels: Optional[int] = None,
        sort_events: bool = True,
        tpc_branches: Optional[List[str]] = None,
        max_events: Optional[int] = None,
        **dataset_kwargs,
    ) -> "SparseWindowDatasetPyG":
        """Load events from ROOT files and build the dataset.

        Events are optionally sorted by (run, subrun, event) before windowing.
        Per-event provenance (run, subrun, event number, source filename) is
        extracted and stored so it can be saved with :meth:`save_events` and
        surfaced per-window via :meth:`window_metadata`.
        """
        import uproot
        import awkward as ak
        from sbn_anomaly.data.materialize_windows import _group_hit_prefixes, _extract_tpc_branch_value

        prefixes = _group_hit_prefixes(hit_branches)
        integral_branches = [p + ".integral" for p in prefixes]
        channel_branches = [p + ".channel" for p in prefixes]
        time_branches = [p + ".time" for p in prefixes]
        wire_branches = [p + ".wire" for p in prefixes]
        plane_branches = [p + ".plane" for p in prefixes]
        tpc_hit_branches = [p + ".tpc" for p in prefixes]
        all_branches = list(set(
            integral_branches + channel_branches +
            time_branches + wire_branches + plane_branches + tpc_hit_branches
        ))
        if tpc_branches:
            all_branches.extend(tpc_branches)

        file_list = list(root_files) if not isinstance(root_files, list) else root_files
        filenames = [str(p) for p in file_list]
        raw_events: list[dict] = []
        discovered_max = -1

        logger.info("Streaming events from %d ROOT file(s)...", len(file_list))
        for file_idx, path in enumerate(file_list):
            try:
                with uproot.open(str(path)) as root_file:
                    if tree_name not in root_file:
                        logger.warning("Tree '%s' not found in %s – skipping.", tree_name, path)
                        continue
                    tree = root_file[tree_name]
                    logger.info("  [%d/%d] %s — %d entries", file_idx + 1, len(file_list), path, tree.num_entries)
                    for chunk in tree.iterate(all_branches, step_size=512, library="ak"):
                        for i in range(len(chunk)):
                            if max_events is not None and len(raw_events) >= max_events:
                                break

                            ch_parts, val_parts = [], []
                            time_parts, wire_parts, plane_parts, tpc_parts = [], [], [], []
                            for integ_b, ch_b, time_b, wire_b, plane_b, tpc_b in zip(
                                integral_branches, channel_branches,
                                time_branches, wire_branches, plane_branches, tpc_hit_branches,
                            ):
                                try:
                                    integrals = ak.to_numpy(ak.flatten(chunk[integ_b][i], axis=None)).astype(np.float32)
                                    channels = ak.to_numpy(ak.flatten(chunk[ch_b][i], axis=None)).astype(np.int64)
                                    times = ak.to_numpy(ak.flatten(chunk[time_b][i], axis=None)).astype(np.float32)
                                    wires = ak.to_numpy(ak.flatten(chunk[wire_b][i], axis=None)).astype(np.int32)
                                    planes = ak.to_numpy(ak.flatten(chunk[plane_b][i], axis=None)).astype(np.int32)
                                    tpcs = ak.to_numpy(ak.flatten(chunk[tpc_b][i], axis=None)).astype(np.int32)
                                except Exception:
                                    continue
                                m = min(len(integrals), len(channels), len(times), len(wires), len(planes), len(tpcs))
                                if m == 0:
                                    continue
                                valid = channels[:m] >= 0
                                ch_parts.append(channels[:m][valid])
                                val_parts.append(integrals[:m][valid])
                                time_parts.append(times[:m][valid])
                                wire_parts.append(wires[:m][valid])
                                plane_parts.append(planes[:m][valid])
                                tpc_parts.append(tpcs[:m][valid])

                            ch_arr = np.concatenate(ch_parts) if ch_parts else np.empty(0, dtype=np.int64)
                            val_arr = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float32)
                            time_arr = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=np.float32)
                            wire_arr = np.concatenate(wire_parts) if wire_parts else np.empty(0, dtype=np.int32)
                            plane_arr = np.concatenate(plane_parts) if plane_parts else np.empty(0, dtype=np.int32)
                            tpc_arr = np.concatenate(tpc_parts) if tpc_parts else np.empty(0, dtype=np.int32)
                            if ch_arr.size:
                                discovered_max = max(discovered_max, int(ch_arr.max()))

                            meta: dict = {}
                            if tpc_branches:
                                for b in tpc_branches:
                                    v = _extract_tpc_branch_value(chunk, i, b)
                                    if v is not None:
                                        meta[b] = v

                            raw_events.append({
                                "channels": ch_arr,
                                "integrals": val_arr,
                                "times": time_arr,
                                "wires": wire_arr,
                                "planes": plane_arr,
                                "tpcs": tpc_arr,
                                "meta": meta,
                                "file_idx": file_idx,
                            })

                        if max_events is not None and len(raw_events) >= max_events:
                            break
            except FileNotFoundError:
                logger.error("ROOT file not found: %s", path)
                raise
            except OSError as exc:
                logger.warning("Skipping unreadable ROOT file %s: %s", path, exc)
                continue

            if max_events is not None and len(raw_events) >= max_events:
                break

        logger.info("Loaded %d events. Sorting: %s", len(raw_events), sort_events)

        if sort_events and tpc_branches:
            raw_events.sort(key=lambda e: _run_subrun_evt_key(e["meta"], tpc_branches))

        ch_flat = np.concatenate([e["channels"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.int64)
        val_flat = np.concatenate([e["integrals"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.float32)
        time_flat = np.concatenate([e["times"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.float32)
        wire_flat = np.concatenate([e["wires"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.int32)
        plane_flat = np.concatenate([e["planes"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.int32)
        tpc_flat = np.concatenate([e["tpcs"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.int32)
        sizes = np.array([len(e["channels"]) for e in raw_events], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)])

        n_ch = n_channels if n_channels is not None else (discovered_max + 1 if discovered_max >= 0 else 0)
        logger.info(
            "Events packed: %d total hits, n_channels=%d  (~%.1f MB)",
            len(ch_flat), n_ch,
            (ch_flat.nbytes + val_flat.nbytes + time_flat.nbytes + wire_flat.nbytes + plane_flat.nbytes + tpc_flat.nbytes + offsets.nbytes) / 1e6,
        )

        # Extract per-event provenance arrays from sorted raw_events
        evt_run, evt_subrun, evt_num = _extract_provenance(raw_events, tpc_branches or [])
        evt_file_idx = np.array([e["file_idx"] for e in raw_events], dtype=np.int32) if raw_events else np.empty(0, dtype=np.int32)

        return cls(
            ch_flat, val_flat, offsets, n_ch,
            evt_run=evt_run, evt_subrun=evt_subrun, evt_num=evt_num,
            evt_file_idx=evt_file_idx, filenames=filenames,
            times_flat=time_flat, wires_flat=wire_flat,
            planes_flat=plane_flat, tpcs_flat=tpc_flat,
            **dataset_kwargs,
        )

    @classmethod
    def from_npz(cls, events_path: str, **dataset_kwargs) -> "SparseWindowDatasetPyG":
        """Load a compact events file saved with :meth:`save_events`.

        Provenance arrays (evt_run, evt_subrun, evt_num, evt_file_idx, filenames)
        are loaded automatically when present; older files without them load fine.
        """
        data = np.load(events_path, allow_pickle=False)
        n_channels = int(data["n_channels"])
        meta_kwargs: dict = {}
        for key in ("evt_run", "evt_subrun", "evt_num", "evt_file_idx"):
            if key in data:
                meta_kwargs[key] = data[key]
        if "filenames" in data:
            meta_kwargs["filenames"] = [str(f) for f in data["filenames"]]
        for key in ("times_flat", "wires_flat", "planes_flat", "tpcs_flat"):
            if key in data:
                meta_kwargs[key] = data[key]
        return cls(
            channels_flat=data["channels_flat"],
            integrals_flat=data["integrals_flat"],
            offsets=data["offsets"],
            n_channels=n_channels,
            **meta_kwargs,
            **dataset_kwargs,
        )

    def save_events(self, path: str) -> None:
        """Save compact sparse events (and provenance metadata if present) to an npz file."""
        arrays: dict = {
            "channels_flat": self._channels_flat,
            "integrals_flat": self._integrals_flat,
            "offsets": self._offsets,
            "n_channels": np.array(self.num_nodes, dtype=np.int64),
        }
        if self._times_flat is not None:
            arrays["times_flat"] = self._times_flat
        if self._wires_flat is not None:
            arrays["wires_flat"] = self._wires_flat
        if self._planes_flat is not None:
            arrays["planes_flat"] = self._planes_flat
        if self._tpcs_flat is not None:
            arrays["tpcs_flat"] = self._tpcs_flat
        if self._evt_run is not None:
            arrays["evt_run"] = self._evt_run
            arrays["evt_subrun"] = self._evt_subrun
            arrays["evt_num"] = self._evt_num
        if self._evt_file_idx is not None:
            arrays["evt_file_idx"] = self._evt_file_idx
        if self._filenames:
            arrays["filenames"] = np.array(self._filenames, dtype="U512")
        np.savez_compressed(path, **arrays)
        has_meta = self._evt_run is not None
        has_hit_geo = self._times_flat is not None
        logger.info(
            "Saved sparse events to %s%s%s", path,
            " (with provenance metadata)" if has_meta else "",
            " (with hit geometry)" if has_hit_geo else "",
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Data:
        start = self._starts[idx]

        frames = []
        for w in range(self.history + 1):
            ws = start + w * self.window_size
            frame = self._compute_frame(ws, ws + self.window_size)
            frames.append(frame.reshape(self.num_nodes, -1))

        past_flat = torch.from_numpy(np.concatenate(frames[:-1], axis=1)).float()
        target_flat = torch.from_numpy(frames[-1]).float()
        x = torch.cat([self._channel_idx, past_flat], dim=1)

        if self.prune_inactive:
            activity = torch.abs(past_flat).sum(dim=1)
            active_idx = torch.where(activity > 1e-6)[0]
            if active_idx.numel() == 0:
                active_idx = torch.zeros(1, dtype=torch.long)
            return self._make_pruned_data(x, target_flat, active_idx)
        return Data(x=x, y=target_flat, edge_index=self.edge_index_full)

    def window_metadata(self, indices=None) -> dict:
        """Return per-window provenance for the given window indices (default: all).

        Returns an empty dict if the dataset was loaded without provenance data
        (i.e. from an events NPZ that pre-dates metadata support).

        Keys present when provenance is available
        -----------------------------------------
        first_run, first_subrun, first_event_num : (N,) int32
            Run/subrun/event number of the first event in each window.
        last_event_num : (N,) int32
            Event number of the last event in each window.
        first_file_idx, last_file_idx : (N,) int32
            Index into ``filenames`` for the first and last event.
        filenames : list[str]
            Ordered list of source ROOT filenames.
        """
        if self._evt_run is None:
            return {}

        starts = np.array(self._starts, dtype=np.int64)
        if indices is not None:
            starts = starts[np.asarray(indices)]

        ends = starts + self.events_per_sample - 1  # inclusive last event index

        result: dict = {
            "first_run":       self._evt_run[starts],
            "first_subrun":    self._evt_subrun[starts],
            "first_event_num": self._evt_num[starts],
            "last_event_num":  self._evt_num[ends],
            "filenames":       list(self._filenames),
        }
        if self._evt_file_idx is not None:
            result["first_file_idx"] = self._evt_file_idx[starts]
            result["last_file_idx"]  = self._evt_file_idx[ends]
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_event_aggregates(self) -> None:
        """Pre-aggregate raw hits per channel for each event into CSR format.

        Stores sorted unique (channel, [sum, count, min, max, sum_sq]) arrays
        so _compute_frame can use np.bincount instead of sorting raw hits on
        every sample access.  columns: 0=sum 1=count 2=min 3=max 4=sum_sq
        """
        n_events = len(self._offsets) - 1
        ch_parts: list[np.ndarray] = []
        feat_parts: list[np.ndarray] = []
        agg_offsets = np.zeros(n_events + 1, dtype=np.int64)

        for i in range(n_events):
            h_start = int(self._offsets[i])
            h_end = int(self._offsets[i + 1])
            ch = self._channels_flat[h_start:h_end]
            val = self._integrals_flat[h_start:h_end]

            mask = ch < self.num_nodes
            ch = ch[mask]
            val = val[mask]

            if ch.size == 0:
                agg_offsets[i + 1] = agg_offsets[i]
                continue

            order = np.argsort(ch, kind="stable")
            ch_s = ch[order]
            val_s = val[order]
            unique_chs, starts, counts = np.unique(ch_s, return_index=True, return_counts=True)

            feats = np.empty((len(unique_chs), 5), dtype=np.float32)
            feats[:, 0] = np.add.reduceat(val_s, starts)           # sum
            feats[:, 1] = counts.astype(np.float32)                # count
            feats[:, 2] = np.minimum.reduceat(val_s, starts)       # min
            feats[:, 3] = np.maximum.reduceat(val_s, starts)       # max
            feats[:, 4] = np.add.reduceat(val_s * val_s, starts)   # sum_sq

            ch_parts.append(unique_chs)
            feat_parts.append(feats)
            agg_offsets[i + 1] = agg_offsets[i] + len(unique_chs)

        self._agg_ch_flat = (
            np.concatenate(ch_parts) if ch_parts else np.empty(0, dtype=np.int64)
        )
        self._agg_feat_flat = (
            np.concatenate(feat_parts) if feat_parts else np.empty((0, 5), dtype=np.float32)
        )
        self._agg_offsets = agg_offsets

    def _compute_frame(self, evt_start: int, evt_end: int) -> np.ndarray:
        """Aggregate events[evt_start:evt_end] into (n_channels, n_bins, n_features).

        Uses pre-aggregated per-event CSR data so each bin only needs
        np.bincount / np.minimum.at on ~events_per_bin small arrays rather
        than sorting all raw hits.
        """
        N = self.num_nodes
        frame = np.zeros((N, self.n_bins, self.n_node_features), dtype=np.float32)

        need_min = "min" in self.node_features
        need_max = "max" in self.node_features
        need_stdev = "stdev" in self.node_features

        for b_idx, split in enumerate(self._bin_splits):
            ch_parts = []
            feat_parts = []
            for i in split:
                e = evt_start + int(i)
                a_start = int(self._agg_offsets[e])
                a_end = int(self._agg_offsets[e + 1])
                if a_end > a_start:
                    ch_parts.append(self._agg_ch_flat[a_start:a_end])
                    feat_parts.append(self._agg_feat_flat[a_start:a_end])

            if not ch_parts:
                continue

            ch = np.concatenate(ch_parts)
            feat = np.concatenate(feat_parts)  # (n_hits, 5)

            # Accumulate sum/count/sum_sq via bincount (O(n + N), no sort needed)
            sums = np.bincount(ch, weights=feat[:, 0], minlength=N).astype(np.float32)
            counts = np.bincount(ch, weights=feat[:, 1], minlength=N).astype(np.float32)
            if need_stdev:
                sum_sq = np.bincount(ch, weights=feat[:, 4], minlength=N).astype(np.float32)

            # min/max via scatter (O(n), no sort needed)
            if need_min:
                mins = np.full(N, np.inf, dtype=np.float32)
                np.minimum.at(mins, ch, feat[:, 2])
            if need_max:
                maxs = np.full(N, -np.inf, dtype=np.float32)
                np.maximum.at(maxs, ch, feat[:, 3])

            for fi, fname in enumerate(self.node_features):
                if fname == "sum":
                    frame[:, b_idx, fi] = sums
                elif fname == "count":
                    frame[:, b_idx, fi] = counts
                elif fname == "mean":
                    frame[:, b_idx, fi] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
                elif fname == "min":
                    frame[:, b_idx, fi] = np.where(np.isfinite(mins), mins, 0.0)
                elif fname == "max":
                    frame[:, b_idx, fi] = np.where(np.isfinite(maxs), maxs, 0.0)
                elif fname == "stdev":
                    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
                    var = np.where(counts > 0, sum_sq / np.maximum(counts, 1) - mean ** 2, 0.0)
                    frame[:, b_idx, fi] = np.sqrt(np.maximum(var, 0.0))

        return frame

    def _make_pruned_data(
        self, x: torch.Tensor, y: torch.Tensor, active_idx: torch.Tensor
    ) -> Data:
        m = active_idx.numel()
        active_np = active_idx.numpy()
        x_pruned = x[active_idx]
        y_pruned = y[active_idx]
        x_pruned[:, 0] = torch.arange(m, dtype=torch.float32) / max(1, m - 1)

        remap = np.full(self.num_nodes, -1, dtype=np.int32)
        remap[active_np] = np.arange(m, dtype=np.int32)
        new_src = remap[self._edge_src_np]
        new_dst = remap[self._edge_dst_np]
        keep = (new_src >= 0) & (new_dst >= 0)
        if keep.any():
            edge_index = torch.from_numpy(
                np.stack([new_src[keep], new_dst[keep]]).astype(np.int64)
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x_pruned,
            y=y_pruned,
            edge_index=edge_index,
            active_mask=active_idx,
            num_nodes_original=self.num_nodes,
        )

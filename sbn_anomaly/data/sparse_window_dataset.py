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
    ) -> None:
        self._channels_flat = np.asarray(channels_flat, dtype=np.int64)
        self._integrals_flat = np.asarray(integrals_flat, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.int64)

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
        Pass ``sort_events=False`` if files are already in temporal order.
        """
        import awkward as ak
        from sbn_anomaly.data.materialize_windows import _group_hit_prefixes, _extract_tpc_branch_value
        from sbn_anomaly.data.streaming import RootStreamer

        prefixes = _group_hit_prefixes(hit_branches)
        integral_branches = [p + ".integral" for p in prefixes]
        channel_branches = [p + ".channel" for p in prefixes]
        all_branches = list(set(integral_branches + channel_branches))
        if tpc_branches:
            all_branches.extend(tpc_branches)

        streamer = RootStreamer(
            file_paths=root_files,
            tree_name=tree_name,
            branches=all_branches,
            batch_size=512,
        )

        raw_events: list[dict] = []
        discovered_max = -1

        logger.info("Streaming events from %d ROOT file(s)...", len(list(root_files)) if isinstance(root_files, list) else 1)
        for batch in streamer.stream():
            for i in range(len(batch)):
                if max_events is not None and len(raw_events) >= max_events:
                    break

                ch_parts, val_parts = [], []
                for integ_b, ch_b in zip(integral_branches, channel_branches):
                    try:
                        integrals = ak.to_numpy(ak.flatten(batch[integ_b][i], axis=None)).astype(np.float32)
                        channels = ak.to_numpy(ak.flatten(batch[ch_b][i], axis=None)).astype(np.int64)
                    except Exception:
                        continue
                    m = min(len(integrals), len(channels))
                    if m == 0:
                        continue
                    valid = channels[:m] >= 0
                    ch_parts.append(channels[:m][valid])
                    val_parts.append(integrals[:m][valid])

                ch_arr = np.concatenate(ch_parts) if ch_parts else np.empty(0, dtype=np.int64)
                val_arr = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float32)
                if ch_arr.size:
                    discovered_max = max(discovered_max, int(ch_arr.max()))

                meta: dict = {}
                if tpc_branches:
                    for b in tpc_branches:
                        v = _extract_tpc_branch_value(batch, i, b)
                        if v is not None:
                            meta[b] = v

                raw_events.append({"channels": ch_arr, "integrals": val_arr, "meta": meta})

            if max_events is not None and len(raw_events) >= max_events:
                break

        logger.info("Loaded %d events. Sorting: %s", len(raw_events), sort_events)

        if sort_events and tpc_branches:
            def _sort_key(evt):
                m = evt["meta"]
                run = subrun = evtnum = None
                for b in tpc_branches:
                    if b in m:
                        if "run" in b.lower() and "subrun" not in b.lower():
                            run = int(m[b])
                        elif "subrun" in b.lower():
                            subrun = int(m[b])
                        elif "evt" in b.lower():
                            evtnum = int(m[b])
                return (run or 999999999, subrun or 999999999, evtnum or 999999999)
            raw_events.sort(key=_sort_key)

        ch_flat = np.concatenate([e["channels"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.int64)
        val_flat = np.concatenate([e["integrals"] for e in raw_events]) if raw_events else np.empty(0, dtype=np.float32)
        sizes = np.array([len(e["channels"]) for e in raw_events], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)])

        n_ch = n_channels if n_channels is not None else (discovered_max + 1 if discovered_max >= 0 else 0)
        logger.info(
            "Events packed: %d total hits, n_channels=%d  (~%.1f MB)",
            len(ch_flat), n_ch,
            (ch_flat.nbytes + val_flat.nbytes + offsets.nbytes) / 1e6,
        )
        return cls(ch_flat, val_flat, offsets, n_ch, **dataset_kwargs)

    @classmethod
    def from_npz(cls, events_path: str, **dataset_kwargs) -> "SparseWindowDatasetPyG":
        """Load a compact events file saved with :meth:`save_events`."""
        data = np.load(events_path, allow_pickle=False)
        n_channels = int(data["n_channels"])
        return cls(
            channels_flat=data["channels_flat"],
            integrals_flat=data["integrals_flat"],
            offsets=data["offsets"],
            n_channels=n_channels,
            **dataset_kwargs,
        )

    def save_events(self, path: str) -> None:
        """Save compact sparse events to an npz file for fast reloading."""
        np.savez_compressed(
            path,
            channels_flat=self._channels_flat,
            integrals_flat=self._integrals_flat,
            offsets=self._offsets,
            n_channels=np.array(self.num_nodes, dtype=np.int64),
        )
        logger.info("Saved sparse events to %s", path)

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

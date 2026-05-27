from __future__ import annotations

from typing import List, Tuple, Optional

import torch


def graph_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_value: float = 0.0, max_nodes: Optional[int] = None):
    """Collate a batch of graph window samples with per-sample node pruning.

    Each item in `batch` is a tuple: (past, adj, target)
      - past: Tensor (T, N, F)
      - adj: Tensor (N, N)
      - target: Tensor (N, F)

    This function:
      - computes per-sample active nodes (non-zero across past/target)
      - extracts only active nodes per sample
      - pads each sample to the batch max active nodes (or `max_nodes` cap)
      - returns (past_batch, adj_batch, target_batch)

    Returns:
      past_batch: Tensor (B, T, M, F)
      adj_batch: Tensor (B, M, M)
      target_batch: Tensor (B, M, F)
    where M = min(max active nodes in batch, max_nodes if provided)
    """
    # Determine per-sample active node indices
    per_sample_selected = []
    T = None
    F = None
    per_sample_total_nodes = []
    
    for past, adj, target in batch:
        # past: (T, N, F)
        total_nodes = int(past.shape[1])
        per_sample_total_nodes.append(total_nodes)
        
        if T is None:
            T = int(past.shape[0])
        if F is None:
            F = int(target.shape[1])
        # activity from past and target
        activity = torch.sum(torch.abs(past), dim=(0, 2))  # (N,)
        activity = activity + torch.sum(torch.abs(target), dim=1)
        sel = torch.nonzero(activity > 0.0, as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            # if no active nodes, pick a single dummy node (index 0)
            sel = torch.tensor([0], dtype=torch.long)
        per_sample_selected.append(sel)

    max_active = max(sel.numel() for sel in per_sample_selected)
    if max_nodes is not None:
        max_active = min(max_active, int(max_nodes))

    B = len(batch)
    M = int(max_active)
    
    # Feature dimension is now F + 1 (add channel index)
    F_extended = F + 1

    past_batch = torch.full((B, T, M, F_extended), pad_value, dtype=batch[0][0].dtype)
    adj_batch = torch.zeros((B, M, M), dtype=batch[0][1].dtype)
    target_batch = torch.full((B, M, F_extended), pad_value, dtype=batch[0][2].dtype)

    for i, (past, adj, target) in enumerate(batch):
        sel = per_sample_selected[i]
        total_nodes = per_sample_total_nodes[i]
        
        if sel.numel() > M:
            sel = sel[:M]
        m = int(sel.numel())
        
        # gather nodes (only active nodes)
        past_sel = past[:, sel, :].contiguous()  # (T, m, F)
        target_sel = target[sel, :].contiguous()  # (m, F)
        adj_sel = adj[sel][:, sel].contiguous()  # (m, m)
        
        # Compute normalized channel indices for these selected nodes
        # sel contains the original channel indices (0 to total_nodes-1)
        # Normalize to 0-1 range
        channel_indices = sel.float() / float(max(total_nodes - 1, 1))  # (m,)
        
        # Prepend channel index to features
        # past_sel_extended: (T, m, F+1)
        past_sel_extended = torch.cat([
            channel_indices.unsqueeze(0).unsqueeze(-1).expand(T, m, 1).to(past_sel.dtype),
            past_sel
        ], dim=-1)
        
        # target_sel_extended: (m, F+1)
        target_sel_extended = torch.cat([
            channel_indices.unsqueeze(-1).to(target_sel.dtype),
            target_sel
        ], dim=-1)
        
        # Copy into batch tensors
        past_batch[i, :, :m, :] = past_sel_extended
        target_batch[i, :m, :] = target_sel_extended
        adj_batch[i, :m, :m] = adj_sel

    return past_batch, adj_batch, target_batch

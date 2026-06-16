"""Pre-processing for raw TPC ADC waveforms.

These transforms run on dense per-event waveform matrices of shape
``(n_channels, n_ticks)`` before they are fed to the waveform VAE.  Everything
here is pure NumPy so it can be unit-tested without torch/ROOT and reused both
in offline materialisation and in a real-time DQM loop.

Typical pipeline::

    wf = pedestal_subtract(adc, pedestal)          # remove per-channel baseline
    wf = remove_coherent_noise(wf, group_size=64)  # subtract common-mode noise
    wf = scale_waveforms(wf, scale=64.0)           # bring into ~[-1, 1]
    wf = pad_or_truncate(wf, n_ticks=4096)         # fix length for the conv VAE
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "pedestal_subtract",
    "remove_coherent_noise",
    "scale_waveforms",
    "pad_or_truncate",
    "preprocess_event",
]


def pedestal_subtract(
    adc: np.ndarray,
    pedestal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Subtract the per-channel baseline.

    Parameters
    ----------
    adc:
        ``(n_channels, n_ticks)`` raw ADC values.
    pedestal:
        ``(n_channels,)`` pedestal per channel (e.g. ``RawDigit.GetPedestal()``).
        If ``None``, the per-channel **median** over ticks is used, which is a
        robust baseline estimate that ignores signal hits.

    Returns
    -------
    ``(n_channels, n_ticks)`` float32 baseline-subtracted waveforms.
    """
    adc = np.asarray(adc, dtype=np.float32)
    if adc.ndim != 2:
        raise ValueError(f"adc must be 2-D (n_channels, n_ticks), got {adc.ndim}-D")
    if pedestal is None:
        base = np.median(adc, axis=1, keepdims=True)
    else:
        base = np.asarray(pedestal, dtype=np.float32).reshape(-1, 1)
        if base.shape[0] != adc.shape[0]:
            raise ValueError(
                f"pedestal length {base.shape[0]} != n_channels {adc.shape[0]}"
            )
    return (adc - base).astype(np.float32)


def remove_coherent_noise(
    wf: np.ndarray,
    group_size: int = 64,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove common-mode (coherent) noise shared by groups of channels.

    Channels read out by the same cold-electronics motherboard / FEMB share a
    correlated noise component.  Subtracting the per-tick **median** across each
    group removes it while leaving real, localized signals (which only appear on
    a few channels) intact.  Median is used rather than mean so a handful of hit
    channels do not bias the estimate.

    Parameters
    ----------
    wf:
        ``(n_channels, n_ticks)`` baseline-subtracted waveforms.
    group_size:
        Number of consecutive channels per coherent-noise group.  Used only when
        ``groups`` is not given.
    groups:
        Optional ``(n_channels,)`` integer array assigning each channel to a
        group (e.g. derived from the FEMB/motherboard map).  Overrides
        ``group_size`` when provided.

    Returns
    -------
    ``(n_channels, n_ticks)`` float32 waveforms with common mode removed.
    """
    wf = np.asarray(wf, dtype=np.float32)
    if wf.ndim != 2:
        raise ValueError(f"wf must be 2-D (n_channels, n_ticks), got {wf.ndim}-D")
    n_chan = wf.shape[0]

    if groups is None:
        if group_size <= 0:
            return wf.copy()
        groups = np.arange(n_chan) // int(group_size)
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != n_chan:
            raise ValueError(
                f"groups length {groups.shape[0]} != n_channels {n_chan}"
            )

    out = wf.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size == 0:
            continue
        common = np.median(wf[idx], axis=0, keepdims=True)  # (1, n_ticks)
        out[idx] = wf[idx] - common
    return out.astype(np.float32)


def scale_waveforms(wf: np.ndarray, scale: float = 64.0) -> np.ndarray:
    """Divide by a fixed ADC scale to bring values into roughly ``[-1, 1]``.

    A *fixed* scale (rather than per-waveform normalisation) is deliberate: it
    keeps amplitude information, so a channel that is abnormally hot or dead is
    still distinguishable after compression.  ``64`` ADC counts is a reasonable
    default for ~few-LSB noise plus O(100)-count signals; tune per detector.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    return (np.asarray(wf, dtype=np.float32) / float(scale)).astype(np.float32)


def pad_or_truncate(wf: np.ndarray, n_ticks: int) -> np.ndarray:
    """Force the tick axis to exactly ``n_ticks`` (pad with zeros / truncate).

    The conv VAE needs a fixed input length divisible by ``2**depth``.
    """
    wf = np.asarray(wf, dtype=np.float32)
    if wf.ndim != 2:
        raise ValueError(f"wf must be 2-D (n_channels, n_ticks), got {wf.ndim}-D")
    cur = wf.shape[1]
    if cur == n_ticks:
        return wf
    if cur > n_ticks:
        return wf[:, :n_ticks].copy()
    pad = np.zeros((wf.shape[0], n_ticks - cur), dtype=np.float32)
    return np.concatenate([wf, pad], axis=1)


def preprocess_event(
    adc: np.ndarray,
    pedestal: Optional[np.ndarray] = None,
    *,
    subtract_pedestal: bool = True,
    pedestal_mode: str = "stored",
    coherent_group_size: int = 64,
    coherent_groups: Optional[np.ndarray] = None,
    remove_coherent: bool = True,
    scale: float = 64.0,
    n_ticks: Optional[int] = None,
) -> np.ndarray:
    """Full preprocessing for one event's ``(n_channels, n_ticks)`` ADC matrix.

    Order: pedestal subtraction -> coherent-noise removal -> scaling -> length fix.
    Each step is individually controllable so callers can expose them in config.

    Parameters
    ----------
    subtract_pedestal:
        If False, skip baseline subtraction entirely (store raw ADC, only
        scaled/length-fixed).
    pedestal_mode:
        ``"stored"`` uses the per-channel ``pedestal`` passed in (e.g.
        ``RawDigit.GetPedestal()``); ``"median"`` ignores it and uses the robust
        per-channel median over ticks instead.
    coherent_groups:
        Optional ``(n_channels,)`` group-id array (e.g. from the electronics
        map) overriding the positional ``coherent_group_size`` blocks.
    remove_coherent, coherent_group_size, scale, n_ticks:
        See the individual transform functions.
    """
    if pedestal_mode not in ("stored", "median"):
        raise ValueError(f"pedestal_mode must be 'stored' or 'median', got {pedestal_mode!r}")

    if subtract_pedestal:
        ped = pedestal if pedestal_mode == "stored" else None
        wf = pedestal_subtract(adc, ped)
    else:
        wf = np.asarray(adc, dtype=np.float32)

    if remove_coherent:
        wf = remove_coherent_noise(
            wf, group_size=coherent_group_size, groups=coherent_groups
        )
    wf = scale_waveforms(wf, scale=scale)
    if n_ticks is not None:
        wf = pad_or_truncate(wf, n_ticks)
    return wf

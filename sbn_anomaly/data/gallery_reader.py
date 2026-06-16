"""Gallery-based raw::RawDigit reader (PyROOT) for use on a LArSoft node.

This reads raw ADC waveforms straight from art/LArSoft files using ``gallery``
through PyROOT -- the same access the C++ macro uses (``Channel()``,
``Samples()``, ``GetPedestal()``, ``ADC()``) -- and yields the same
:class:`~sbn_anomaly.data.raw_digit_reader.RawEvent` objects the rest of the
pipeline consumes. That lets you go straight from an art file to the training
``.npz`` in one step, with no intermediate flat ntuple.

Requirements: a set-up LArSoft/experiment environment (gpvm or container) where
``gallery`` and ``lardataobj`` dictionaries are loadable. ROOT is imported
lazily so importing this module does nothing on a plain laptop.

Example (one-step waveform npz)::

    python scripts/rawdigits_to_npz_gallery.py \
        --config configs/raw_vae.yaml \
        --files /exp/icarus/data/.../raw_decoded_reco_094.root \
        --output data/raw_waveforms_train.npz --max-waveforms 2000000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

from sbn_anomaly.data.raw_digit_reader import RawEvent

logger = logging.getLogger(__name__)


def _vec_to_int16(vec, np_mod) -> np.ndarray:
    """Convert a std::vector<short> to a numpy int16 array efficiently."""
    n = int(vec.size())
    try:
        # cppyy exposes .data() with the buffer protocol; frombuffer avoids a
        # Python-level loop over thousands of ticks.
        buf = vec.data()
        arr = np_mod.frombuffer(buf, dtype=np_mod.int16, count=n)
        return arr.copy()
    except Exception:
        return np_mod.fromiter((vec[i] for i in range(n)), dtype=np_mod.int16, count=n)


class GalleryRawDigitReader:
    """Iterate events from art files via gallery, yielding :class:`RawEvent`.

    Parameters
    ----------
    file_paths:
        art ROOT file path(s) (raw/decoded), e.g. the decoder output.
    tag:
        Input tag / module label for the RawDigit product (``"daq"``).
    max_events:
        Stop after this many events (``None`` = all).
    uncompress:
        If True (default) and a digit reports a non-zero compression, expand it
        with ``raw::Uncompress`` before reading. Decoded data is normally
        uncompressed, so this is usually a no-op.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tag: str = "daq",
        max_events: Optional[int] = None,
        uncompress: bool = True,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths: List[str] = [str(p) for p in file_paths]
        self.tag = tag
        self.max_events = max_events
        self.uncompress = bool(uncompress)

    def __iter__(self) -> Iterator[RawEvent]:
        import ROOT  # lazy: only available on a LArSoft node

        ROOT.gErrorIgnoreLevel = ROOT.kError
        # Force-load the gallery + lardataobj dictionaries. With a set-up
        # environment ROOT autoloads via rootmaps, but loading explicitly makes
        # failures obvious and early.
        for lib in ("libgallery", "liblardataobj_RawData"):
            try:
                ROOT.gSystem.Load(lib)
            except Exception:
                pass

        gallery = ROOT.gallery
        art = ROOT.art
        RawDigitVec = ROOT.std.vector("raw::RawDigit")

        StrVec = ROOT.std.vector(ROOT.std.string)
        files = StrVec()
        for p in self.file_paths:
            files.push_back(p)

        ev = gallery.Event(files)
        tag = art.InputTag(self.tag)
        get_handle = ev.getValidHandle[RawDigitVec]

        n_yielded = 0
        while not ev.atEnd():
            if self.max_events is not None and n_yielded >= self.max_events:
                break
            try:
                digits = get_handle(tag).product()
            except Exception as exc:
                logger.warning("getValidHandle failed on an event: %s", exc)
                ev.next()
                continue

            ndig = int(digits.size())
            if ndig == 0:
                ev.next()
                continue

            aux = ev.eventAuxiliary()
            run, subrun, event = int(aux.run()), int(aux.subRun()), int(aux.event())

            first = digits.at(0)
            nticks = int(first.Samples())

            channel = np.empty(ndig, dtype=np.int32)
            pedestal = np.empty(ndig, dtype=np.float32)
            adc = np.empty((ndig, nticks), dtype=np.float32)

            for i in range(ndig):
                rd = digits.at(i)
                channel[i] = int(rd.Channel())
                pedestal[i] = float(rd.GetPedestal())
                wf = self._read_adc(rd, ROOT, nticks)
                if wf.shape[0] != nticks:
                    fixed = np.full(nticks, pedestal[i], dtype=np.float32)
                    m = min(nticks, wf.shape[0])
                    fixed[:m] = wf[:m]
                    wf = fixed
                adc[i] = wf

            yield RawEvent(
                run=run, subrun=subrun, event=event,
                channel=channel, pedestal=pedestal, adc=adc,
            )
            n_yielded += 1
            ev.next()

    def _read_adc(self, rd, ROOT, nticks: int) -> np.ndarray:
        """Return the (uncompressed) ADC waveform of one RawDigit as float32."""
        comp = 0
        try:
            comp = int(rd.Compression())
        except Exception:
            comp = 0
        if comp != 0 and self.uncompress:
            out = ROOT.std.vector("short")()
            try:
                ROOT.raw.Uncompress(rd.ADCs(), out, rd.Compression())
                return _vec_to_int16(out, np).astype(np.float32)
            except Exception as exc:
                logger.warning("raw::Uncompress failed (comp=%s): %s; "
                               "reading stored ADCs as-is", comp, exc)
        # Uncompressed (or uncompress unavailable): read fADC directly.
        try:
            return _vec_to_int16(rd.ADCs(), np).astype(np.float32)
        except Exception:
            # Last resort: per-tick accessor (slow but always correct).
            n = int(rd.Samples())
            return np.fromiter((rd.ADC(i) for i in range(n)),
                               dtype=np.float32, count=n)

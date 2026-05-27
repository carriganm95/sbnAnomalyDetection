"""Geometry parsing helpers.

These helpers are intentionally small and dependency-free so they can be used to
extract channel/electronics positions from XML-like geometry snippets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class PositionRecord:
    name: str
    x: float
    y: float
    z: float
    plane: int | None = None
    channel: int | None = None
    wire: int | None = None


def _parse_wire_name(name: str) -> tuple[int | None, int | None]:
    """Extract a wire plane index and wire number from names like posWire_U_1."""
    if not name.startswith("posWire_"):
        return None, None

    parts = name.split("_")
    if len(parts) < 3:
        return None, None

    plane_token = parts[1].upper()
    plane_map = {"U": 0, "V": 1, "Y": 2}
    plane_index = plane_map.get(plane_token)

    try:
        wire_number = int(parts[2])
    except ValueError:
        wire_number = None

    return plane_index, wire_number


def _parse_position_element(element: ET.Element) -> PositionRecord:
    name = element.attrib.get("name")
    if not name:
        raise ValueError("position element missing required 'name' attribute")

    try:
        x = float(element.attrib["x"])
        y = float(element.attrib["y"])
        z = float(element.attrib["z"])
    except KeyError as exc:
        raise ValueError(f"position element {name!r} missing required attribute: {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError(f"position element {name!r} has a non-numeric coordinate") from exc

    plane_index, wire_number = _parse_wire_name(name)
    return PositionRecord(
        name=name,
        x=x,
        y=y,
        z=z,
        plane=plane_index,
        channel=wire_number,
        wire=wire_number,
    )


def parse_position_xml(text: str, name_prefix: str | None = None) -> list[PositionRecord]:
    """Parse one or more standalone <position .../> lines from XML text.

    The input does not need to be a full XML document; the function wraps it in a
    synthetic root element first, so it can handle snippets copied directly from
    geometry files.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        wrapped = f"<root>{text}</root>"
        root = ET.fromstring(wrapped)
    records: list[PositionRecord] = []
    for element in root.iter("position"):
        record = _parse_position_element(element)
        if name_prefix is not None and not record.name.startswith(name_prefix):
            continue
        records.append(record)
    return records


def parse_position_file(path: str | Path, name_prefix: str | None = None) -> list[PositionRecord]:
    """Parse all <position .../> entries from a file."""
    text = Path(path).read_text()
    return parse_position_xml(text, name_prefix=name_prefix)


def position_lookup(records: Iterable[PositionRecord]) -> dict[str, PositionRecord]:
    """Build a name -> record lookup for quick coordinate access."""
    return {record.name: record for record in records}

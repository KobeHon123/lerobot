"""Relative XY mapping built from the top-camera thumb/index MCP midpoint."""

from __future__ import annotations


class RelativeXYMapper:
    """Convert a top-camera hand midpoint into relative XY deltas from a baseline.

    This mapper intentionally tracks only relative motion. The baseline is established
    once the user calibrates the neutral hand pose, and later frames report the delta
    from that neutral point. This makes it suitable for mapping hand motion onto a
    simulator top-view XY plane without tying the control signal to absolute camera
    coordinates.
    """

    def __init__(self, deadzone_px: float = 8.0, max_delta_px: float = 160.0) -> None:
        self.deadzone_px = deadzone_px
        self.max_delta_px = max_delta_px
        self._reference: tuple[float, float] | None = None

    def reset(self, midpoint: tuple[float, float]) -> None:
        self._reference = midpoint

    def update(self, midpoint: tuple[float, float]) -> dict[str, float]:
        if self._reference is None:
            self.reset(midpoint)

        ref_x, ref_y = self._reference
        dx_px = midpoint[0] - ref_x
        dy_px = midpoint[1] - ref_y

        if abs(dx_px) < self.deadzone_px:
            dx_px = 0.0
        if abs(dy_px) < self.deadzone_px:
            dy_px = 0.0

        dx_px = max(-self.max_delta_px, min(self.max_delta_px, dx_px))
        dy_px = max(-self.max_delta_px, min(self.max_delta_px, dy_px))

        dx_norm = dx_px / self.max_delta_px if self.max_delta_px > 0 else 0.0
        dy_norm = dy_px / self.max_delta_px if self.max_delta_px > 0 else 0.0

        return {
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "dx_norm": float(dx_norm),
            "dy_norm": float(dy_norm),
            "mid_x": float(midpoint[0]),
            "mid_y": float(midpoint[1]),
        }

    @property
    def reference(self) -> tuple[float, float] | None:
        return self._reference

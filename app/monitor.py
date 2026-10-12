"""Pure region monitoring, no Qt/cv2. Easy to unit-test."""

MARGIN = 40


def center_of(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def lock_region(box, margin=MARGIN, fw=None, fh=None):
    """Expand box into a fixed region. Clamp to frame when size known."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    if fw is not None and fh is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw - 1, x2)
        y2 = min(fh - 1, y2)
    return (x1, y1, x2, y2)


def is_inside(center, rect):
    cx, cy = center
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2


class RegionMonitor:
    """Fixed regions, one per person. ON = lock all, OFF = clear."""

    def __init__(self, margin=MARGIN):
        self.margin = margin
        self.regions = []

    @property
    def active(self):
        return len(self.regions) > 0

    @property
    def region(self):
        """First region (compat for single-person callers)."""
        return self.regions[0] if self.regions else None

    def lock(self, box, fw=None, fh=None):
        rect = lock_region(box, self.margin, fw, fh)
        self.regions = [rect]
        return rect

    def lock_all(self, boxes, fw=None, fh=None):
        self.regions = [lock_region(b, self.margin, fw, fh) for b in boxes]
        return self.regions

    def clear(self):
        self.regions = []

    def check_box(self, box):
        """True when box center is inside region. False when outside."""
        if not self.regions:
            return True
        return is_inside(center_of(box), self.regions[0])

    def check_all(self, boxes):
        """Per-region bool. Region OK when >=1 person center inside.

        # ponytail: no ID tracking, swap-proof only when regions disjoint.
        # Add ByteTrack/SORT when identity must persist across overlap.
        """
        centers = [center_of(b) for b in boxes]
        return [
            any(is_inside(c, r) for c in centers) if centers else False
            for r in self.regions
        ]


def demo():
    m = RegionMonitor(margin=10)
    m.lock((100, 100, 200, 200))
    assert m.check_box((110, 110, 190, 190)) is True
    assert m.check_box((300, 300, 350, 350)) is False
    m.lock_all([(100, 100, 200, 200), (400, 400, 500, 500)])
    assert m.check_all([(110, 110, 190, 190)]) == [True, False]
    assert m.check_all([(110, 110, 190, 190), (410, 410, 490, 490)]) == [True, True]
    assert m.check_all([]) == [False, False]
    m.clear()
    assert m.active is False
    print("monitor ok")


if __name__ == "__main__":
    demo()

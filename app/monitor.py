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
    """One fixed region. ON = lock, OFF = clear."""

    def __init__(self, margin=MARGIN):
        self.margin = margin
        self.region = None

    @property
    def active(self):
        return self.region is not None

    def lock(self, box, fw=None, fh=None):
        self.region = lock_region(box, self.margin, fw, fh)
        return self.region

    def clear(self):
        self.region = None

    def check_box(self, box):
        """True when box center is inside region. False when outside."""
        if self.region is None:
            return True
        return is_inside(center_of(box), self.region)


def demo():
    m = RegionMonitor(margin=10)
    m.lock((100, 100, 200, 200))
    assert m.check_box((110, 110, 190, 190)) is True
    assert m.check_box((300, 300, 350, 350)) is False
    m.clear()
    assert m.active is False
    print("monitor ok")


if __name__ == "__main__":
    demo()

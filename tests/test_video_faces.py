import numpy as np
import pytest

from deepfake_detector.faces import Box, FaceTracker, OpenCVHaarDetector, build_face_detector
from deepfake_detector.video import VideoReadError, iter_frames, plan_clips, probe


def test_probe_and_iter_frames(video):
    info = probe(video)
    assert info.frame_count == 48 and info.width == 160 and info.height == 120
    assert info.duration_s == pytest.approx(2.0, abs=0.1)
    indices = [i for i, _ in iter_frames(video, 10, 15)]
    assert indices == [10, 11, 12, 13, 14]


def test_probe_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoReadError):
        probe(bad)


@pytest.mark.parametrize(
    "frames,clips,span,expected",
    [
        (300, 3, 80, [(0, 100), (100, 200), (200, 300)]),
        (100, 3, 80, [(0, 100)]),  # too short for more than one clip
        (10, 3, 80, [(0, 10)]),
        (0, 3, 80, []),
    ],
)
def test_plan_clips(frames, clips, span, expected):
    assert plan_clips(frames, clips, span) == expected


def test_box_iou():
    a = Box(0, 10, 10, 0)
    assert a.iou(a) == 1.0
    assert a.iou(Box(0, 20, 10, 10)) == 0.0
    assert a.iou(Box(0, 15, 10, 5)) == pytest.approx(50 / 150)


class SeqDetector:
    name = "seq"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def detect(self, frame):
        self.calls += 1
        return self.outputs.pop(0)


def test_tracker_starts_largest_then_follows_subject():
    frame = np.zeros((200, 200, 3), np.uint8)
    small, big = Box(0, 40, 40, 0), Box(100, 190, 190, 100)
    big_moved = Box(105, 195, 195, 105)
    tracker = FaceTracker(SeqDetector([[small, big], [Box(0, 60, 60, 0), big_moved]]))
    assert tracker.update(frame) == big
    # A larger face appears elsewhere, but we keep following the overlapping one.
    assert tracker.update(frame) == Box(105, 195, 195, 105).clip(200, 200)


def test_tracker_detect_every_reuses_box():
    frame = np.zeros((50, 50, 3), np.uint8)
    det = SeqDetector([[Box(0, 10, 10, 0)], [Box(0, 12, 12, 0)]])
    tracker = FaceTracker(det, detect_every=3)
    boxes = [tracker.update(frame) for _ in range(4)]
    assert det.calls == 2 and boxes[1] == boxes[2] == boxes[0]


def test_opencv_backend_runs_on_blank_frame():
    detector = build_face_detector("opencv")
    assert isinstance(detector, OpenCVHaarDetector)
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []

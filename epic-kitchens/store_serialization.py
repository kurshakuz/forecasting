from epic_kitchens.hoa import load_detections, save_detections

from epic_kitchens.hoa.types import (
    FrameDetections,
    ObjectDetection,
    HandDetection,
    FloatVector,
    HandState,
    HandSide,
    BBox,
)

from pathlib import Path

def test_serialisation_round_trip_is_idempotent(tmp_path):
    video_id = "P01_101"
    frame_number = 10

    detection = FrameDetections(
        video_id=video_id,
        frame_number=frame_number,
        objects=[ObjectDetection(bbox=BBox(0.1, 0.2, 0.3, 0.4), score=0.1)],
        hands=[
            HandDetection(
                bbox=BBox(0.2, 0.3, 0.4, 0.5),
                score=0.2,
                state=HandState.PORTABLE_OBJECT,
                side=HandSide.RIGHT,
                object_offset=FloatVector(x=0.1, y=0.1),
            )
        ],
    )

    filepath = tmp_path / (video_id + ".pkl")

    save_detections([detection], filepath)
    loaded_detections = load_detections(filepath)[0]
    print(loaded_detections)

    filepath2 = tmp_path / "P01_01.pkl"
    loaded_detections2 = load_detections(filepath2)[0]
    print(loaded_detections2)


test_serialisation_round_trip_is_idempotent(Path("./temp"))
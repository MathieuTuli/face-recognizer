import argparse
import cv2

from facedetect_v2 import FaceDetector
from video_manager import VideoManager
from yolo_manager import YOLOManager
from components import Frame
from yolo_api import YOLOAPI
from faceid import FaceID

parser = argparse.ArgumentParser()
parser.add_argument("--cam",
                    dest="cam",
                    default=0,
                    type=int)
args = parser.parse_args()


class FaceRecognizer:
    def __init__(self,
                 yolo_manager: YOLOManager,
                 face_detector: FaceDetector,
                 video_manager: VideoManager,
                 face_id: FaceID):
        """
        """
        self.yolo = yolo_manager
        self.face_detect = face_detector
        self.video_manager = video_manager

    def run(self,) -> None:
        while True:
            frame = self.video_manage.get_frame()
            if not isinstance(frame, Frame):
                raise ValueError(f"received frame not of type {type(Frame)}")
            if frame is None:
                continue


if __name__ == "__main__":
    ...

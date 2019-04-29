from typing import Any, Dict, List

import importlib.resources
import threading
import numpy as np
import prctl

# from .facedetect_v2 import FaceDetector
from .face_detector import FaceDetector
from .process_bus import ProcessBus
from .components import Frame, ObjectIdentifierObject, FaceRecognizerOutput, \
    Face, FaceLabel
from .features import Feature
from .drawing import Drawer
from .face_id import FaceID
# from .face_id import FaceID

FACE_DETECTOR_MODEL_FILE = importlib.resources.path(
        'face_recognizer.model_data',
        'face_detector.pb')
FACE_RECOGNIZER_WHO = importlib.resources.read_text(
        'face_recognizer.model_data',
        'face_recognizer_who.txt')
DEFAULT_COCO_CLASSES = importlib.resources.read_text(
        'face_recognizer.model_data',
        'coco_classes.txt').split()
DEFAULT_FACE_ID_MODEL_FILE = importlib.resources.path(
        'face_recognizer.model_data',
        'face_id.pb')


class FaceRecognizer(Feature):
    def __init__(self,
                 name: str,
                 process_bus: ProcessBus,
                 barrier: threading.Barrier,
                 drawer: Drawer,
                 settings: Dict[str, Any]) -> None:
        """
        """
        Feature.__init__(self,
                         name=name,
                         drawer=drawer,
                         barrier=barrier,
                         process_bus=process_bus,
                         draw=settings['draw'])
        self.name = name
        self.labels = None
        self.settings = settings
        self.face_detector = FaceDetector(
                FACE_DETECTOR_MODEL_FILE.gen,
                gpu_frac=settings['face_detector_gpu_frac'])
        self.show_all_labels = settings['show_all_labels']
        self.face_id = FaceID([1, 1],
                              DEFAULT_COCO_CLASSES,
                              # this is a string with names seperated by '_'
                              {'target': FACE_RECOGNIZER_WHO,
                               'gpu_frac': settings['face_id_gpu_frac'],
                               'model_name': DEFAULT_FACE_ID_MODEL_FILE.gen})

    def run(self,) -> None:
        prctl.set_name(str(self))
        if self.in_q is None or self.out_q is None:
            raise ValueError(f"Queues are broken. in_q is {self.in_q} and " +
                             f"out_q is {self.out_q}")
        while True:
            identifier_frame = self.in_q.get_nowait()
            if identifier_frame is None:
                continue
            if identifier_frame.frame is None:
                self.send_results(
                        FaceRecognizerOutput(
                            faces=None,
                            frame=None,
                            overlay=None
                            ))
                self.barrier.wait()
                continue
            if not isinstance(identifier_frame, Frame):
                raise ValueError(f"received frame not of type {type(Frame)}")
            results = self.process_frame(identifier_frame)
            self.send_results(results)
            self.barrier.wait()

    def process_frame(self,
                      frame: Frame,) -> FaceRecognizerOutput:
        faces = self.recognize_faces(frame.frame)
        overlay = self.drawer.draw_faces(frame.frame * 0,
                                         faces) if self.draw else None
        return FaceRecognizerOutput(
                faces=faces,
                frame=frame,
                overlay=overlay,
                )

    def recognize_faces(self,
                        frame: np.ndarray,) -> List[Face]:
        # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, boxes = self.face_detector.detect(frame)
        if faces.any():
            results = self.face_id.identify(
                    faces, boxes,
                    threshold=self.face_id_threshold,
                    show_all_labels=self.show_all_labels)
            # if not isinstance(faces, List[Face]):
            #     raise ValueError(f"{faces} is of incorrect type")
        return results

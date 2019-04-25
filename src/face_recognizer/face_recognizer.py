from typing import Any, Dict, List

import threading
import numpy as np
import prctl

# from .facedetect_v2 import FaceDetector
from .process_bus import ProcessBus
from .components import Frame, FaceRecognizerOutput, Face, FaceLabel
from .features import Feature
from .drawing import Drawer
# from .face_id import FaceID


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
        self.model = None
        self.labels = None

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
                      frame: Frame) -> FaceRecognizerOutput:
        faces = self.recognize_faces(frame.frame)
        overlay = self.draw.draw_faces(
                )
        return FaceRecognizerOutput(
                faces=faces,
                frame=frame,
                overlay=overlay,
                )

    def recognize_faces(self, frame: np.ndarray) -> List[Face]:
        ...

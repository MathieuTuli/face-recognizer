'''YoloManager Help'''

from typing import List, Dict, Any, Union
import importlib.resources
import logging

from PIL import Image as PILImage

from .process_bus import ProcessBus
from .components import Frame, SharedTransceiver, YoloObject, \
        YoloOutput, BBox, MonoQueue, UnlimitedQueue
from .yolo import YOLO
from pydarknet import Detector
from pydarknet import Image as DarkImage
import prctl


logger = logging.getLogger(__name__)


class YOLOManager(SharedTransceiver):
    '''
    YoloManager

    API to various yolo models
    '''
    def __init__(self,
                 process_bus: ProcessBus,
                 class_names: List[str],
                 settings: Dict[str, Any],):
        '''Constructor
        '''
        self.name = settings['name']
        SharedTransceiver.__init__(self, self.name)
        self.class_names = class_names
        self.process_bus = process_bus
        self.threshold = settings['threshold']
        self.yolo_version = settings['version']
        self.class_filters = settings['class_filters']
        self.on = settings['feature_on']
        if self.yolo_version not in ['keras', 'pydarknet']:
            raise ValueError(f"Unknown YOLO version {self.yolo_version}")
        # NOTE the network must be intialized in self.run() for multiprocessing
        # reasons
        self.net: Any = None

    def __str__(self) -> str:
        """Object Identifier"""
        return str(self.name)

    def register_customers(self,
                           in_q_owner: str,
                           in_q_name: str,
                           out_q_owner: str,
                           out_q_name: str,) -> None:
        self.round_robin.append((
            self.process_bus.get_queue(in_q_owner, in_q_name),
            self.process_bus.get_queue(out_q_owner, out_q_name)))

    def get_yolo_results(self,
                         request_frame: Frame) -> YoloOutput:
        '''
        Grab yolo results and return
        Note that all results are normalized to being formatted as follows:
        List[Tuple[class_names, class_score, BBox]]
        '''
        frame = request_frame.frame
        yolo_results = YoloOutput(
                yolo_objects=list(),
                frame=request_frame)
        if self.yolo_version == 'keras':
            pil_frame = PILImage.fromarray(frame)
            original_frame, r_image, \
                boxes, scores, classes = self.net.detect_image(
                        pil_frame,
                        threshold=self.threshold,
                        classes_to_detect=self.class_filters,)
            boxes = boxes.astype(int).tolist()
            if len(boxes) != len(scores) and \
                    len(scores) != len(classes) and \
                    len(boxes) != len(classes):
                raise Exception("YOLO Keras failed in detecting")
            for i, (t, l, b, r) in enumerate(boxes):
                class_name = self.class_names[classes[i]]
                if class_name not in self.class_filters:
                    continue
                yolo_results.yolo_objects.append((YoloObject(
                    class_name=self.class_names[classes[i]],
                    score=scores[i],
                    bbox=BBox(left=int(l),
                              top=int(t),
                              right=int(r),
                              bottom=int(b)))))
        elif self.yolo_version == 'pydarknet':
            dark_frame = DarkImage(frame)
            results = self.net.detect(dark_frame, self.threshold)
            del dark_frame
            yolo_results = YoloOutput(
                    yolo_objects=[YoloObject(
                        class_name=class_name.decode("utf-8"),
                        score=score,
                        bbox=BBox(
                            left=int(x - w/2),
                            top=int(y - h/2),
                            right=int(x + w/2),
                            bottom=int(y + h/2))) for
                            class_name, score, (x, y, w, h) in results
                            if class_name.decode("utf-8") in
                            self.class_filters],
                    frame=request_frame)
        return yolo_results

    def load_model(self) -> None:
        if self.yolo_version == 'keras':
            self.net = YOLO()
        elif self.yolo_version == 'pydarknet':
            cfg_file = str(next(importlib.resources.path(
                    'imrsv.object_tracking.model_data',
                    'yolov3.cfg').gen))
            weights_file = str(next(importlib.resources.path(
                    'imrsv.object_tracking.model_data',
                    'yolov3.weights').gen))
            coco_file = str(next(importlib.resources.path(
                    'imrsv.object_tracking.model_data',
                    'coco.data').gen))
            self.net = Detector(bytes(cfg_file, encoding="utf-8"),
                                bytes(weights_file, encoding="utf-8"),
                                0,
                                bytes(coco_file, encoding="utf-8"))

    def run(self) -> None:
        """
        Process target

        input to yolo from caller will always be a np.ndarray.
        output is formatted as follows:
        list((class: str:,
              score: float,
              box)
        Note: box is normalized to [left, top, right, bototm] coordinates
        """
        prctl.set_name(str(self))
        if not self.on:
            logger.info("YOLO is turned off, returning")
            return
        self.load_model()
        while True:
            # TODO: fix round robin overhead: if a feed is not giving frames,
            # need to not spend time polling it until frames start coming in
            # again. How to do this?
            for in_queue, out_queue in self.round_robin:
                # request is of type Frame
                request_frame = in_queue.get_nowait()
                if request_frame is None:
                    continue
                if not isinstance(request_frame, Frame):
                    raise ValueError(
                            "Yolo received a frame of not type 'Frame'")
                if request_frame.frame is None:
                    # Signal to Tracker that a 'None' frame was received
                    out_queue.put(YoloOutput(
                        yolo_objects=None,
                        frame=None))
                    continue
                yolo_results = self.get_yolo_results(request_frame)
                out_queue.put(yolo_results)


__all__ = 'YOLOManager',

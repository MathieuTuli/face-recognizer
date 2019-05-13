from typing import Dict, Any, List
import threading
import logging

import prctl

from .object_identifier import ObjectIdentifier
from .process_bus import ProcessBus
from .components import ObjectIdentifierOutput, ObjectIdentifierObject, Frame
from .features import Feature
from .drawing import Drawer


logger = logging.getLogger(__name__)


class ObjectIdentifierPassthrough(Feature):
    def __init__(self,
                 name: str,
                 process_bus: ProcessBus,
                 barrier: threading.Barrier,
                 drawer: Drawer,
                 settings: Dict[str, Any]):
        """
        """
        Feature.__init__(self,
                         name=name,
                         drawer=drawer,
                         barrier=barrier,
                         process_bus=process_bus,
                         draw=settings['draw'])
        self.name = name
        self.barrier = barrier
        self.class_filters = settings['class_filters']
        self.settings = settings

    def __str__(self) -> str:
        '''
        '''
        return self.name

    def run(self) -> None:
        prctl.set_name(str(self))
        if self.in_q is None or self.out_q is None:
            raise ValueError("in_q or out_q is None")
        while True:
            # of type ObjectIdentifierOutput
            object_identifier_results = self.in_q.get_nowait()
            if object_identifier_results is None:
                continue
            if not isinstance(object_identifier_results,
                              ObjectIdentifierOutput):
                raise ValueError(
                        "Queue gave values not of type ObjectIdentifierOutput")
            if object_identifier_results.frame is None:
                # signal to depency, if any, that the frame was None
                self.send_results(
                    ObjectIdentifierOutput(
                       objects=None,
                       frame=None,
                       overlay=None,)
                )
                self.barrier.wait()
                continue
            frame = object_identifier_results.frame
            object_identifier_objects = object_identifier_results.objects
            results = self.process_frame(frame, object_identifier_objects)
            self.send_results(results)
            self.barrier.wait()

    def process_frame(self,
                      frame: Frame,
                      object_identifier_objects: List[
                          ObjectIdentifierObject]):
        # TODO class filters
        overlay = self.drawer.draw_bounding_boxes(
                frame.frame,
                object_identifier_objects,
                overlay=True) if self.draw else None
        results = ObjectIdentifierOutput(
                objects=object_identifier_objects,
                frame=frame,
                overlay=overlay)
        return results


__all__ = 'ObjectIdentifierPassthrough',

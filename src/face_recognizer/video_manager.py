'''VideoManager Help'''

from datetime import datetime
from typing import Union, Tuple, List
from signal import signal, SIGINT, SIG_DFL
import subprocess
import threading
import prctl
import math
import os

import psycopg2
import numpy as np
import cv2
import av

from .process_bus import ProcessBus
from .components import Source, Frame, MonoQueue, UnlimitedQueue


class VideoManager(Source):
    ''''Video Manager'''
    def __init__(self,
                 process_bus: ProcessBus,
                 source: Union[int, str, Tuple[str, str]],
                 barrier: threading.Barrier,
                 input_type: str = 'file',):
        '''
        '''
        Source.__init__(self)
        self.out: List[Union[MonoQueue, UnlimitedQueue]] = list()
        self.process_bus = process_bus
        self.barrier = barrier
        self.source: Union[ImageSource, VideoSource]
        if input_type == 'image':
            self.source = ImageSource(source)
        elif input_type in ['webcam', 'video']:
            self.source = VideoSource(source,
                                      input_type)
        else:
            raise ValueError("Why isn't @param 'input_type'one of ['image'," +
                             " 'video', 'webcam']?")

    def register_out(self, owner: str, q_name: str) -> None:
        '''Register a component's queue for frame shunting'''
        self.out.append(self.process_bus.get_queue(owner, q_name))

    def run(self) -> None:
        '''Frame retrieval and id-ing'''
        prctl.set_name(str(self))
        while True:
            frame = self.source.get_frame()
            if not isinstance(frame, Frame):
                raise ValueError("'frames' from source.get_frame() must " +
                                 "of type 'Frame'")
            for q in self.out:
                q.put(frame)
            self.barrier.wait()


class MediaSource:
    '''
    Parent media class
    '''
    def get_frame(self) -> Frame:
        '''
        Must overwrite
        '''
        raise NotImplementedError("Overwrite: frame grabbing function")


class VideoSource(MediaSource):
    '''
    Video source from file or webcam
    '''
    def __init__(self,
                 source: Union[str, int],
                 video_type: str):
        '''
        Constructor
        '''
        if video_type == 'video':
            if not os.path.isfile(source):
                raise FileNotFoundError(f"{source} is not a valid file source")
            self.source = cv2.VideoCapture(source)
            # OpenCV installed handler that ignores sigint: reset to default
            signal(SIGINT, SIG_DFL)
        elif video_type == 'webcam':
            if not isinstance(source, int):
                raise ValueError(f"{source} is not a valid webcam source")
            self.source = cv2.VideoCapture(source)
            # OpenCV installed handler that ignores sigint: reset to default
            signal(SIGINT, SIG_DFL)
        MediaSource.__init__(self)

    def get_frame(self) -> Frame:
        '''
        Grab frame from cv2.VideoCapture source
        '''
        r, frame = self.source.read()
        if not r:
            return Frame(frame=None, timestamp=None)
        return Frame(frame=frame, timestamp=datetime.now())


class ImageSource(MediaSource):
    '''
    Image source from file
    '''
    def __init__(self,
                 source: str):
        '''
        Constructor
        '''
        if not os.path.isfile(source):
            raise FileNotFoundError(f"{source} is not a valid file source")
        self.source = source
        MediaSource.__init__(self)

    def get_frame(self) -> Frame:
        '''
        Grab frame from cv2.imread()
        '''
        frame = cv2.imread(self.source)
        if frame is None:
            return Frame(frame=None, timestamp=None)
        return Frame(frame=frame, timestamp=datetime.now())


__all__ = 'VideoManager',

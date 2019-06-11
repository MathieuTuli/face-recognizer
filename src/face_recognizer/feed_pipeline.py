'''Camera Process Help'''

from typing import Union, Dict, Any, List, Tuple
from abc import abstractmethod
import multiprocessing
import threading
import logging
import os

import prctl
# import cv2

from .video_manager import VideoManager
from .object_identifier import ObjectIdentifier
from .face_recognizer_feature import FaceRecognizer
from .process_bus import ProcessBus
from .data_sink import DataSink
from .drawing import Drawer


class FeedPipeline:
    """Camera Process
    Spawned in main | Each camera runs as its own process, and here is
    where managing camera related tasks occurs, from receiving a frame
    and processing it to saving data.
    """
    def __init__(self,
                 name: str,
                 feed_type: str,
                 source: Union[int, str],
                 process_bus: ProcessBus,
                 object_identifier: ObjectIdentifier,
                 drawer: Drawer,
                 settings: Dict[str, Any],):
        '''
        Constructor
        :param str name: unique identifier for this pipeline
        :param str feed_type: options = 'webcam', 'video', 'image', 'webrtc'
        :param source: video source to grab a frame from.
        :type source: int or str or multiprocessing.Queue()
        :param ProcessBus process_bus: Queue map object for shared queue access
        :param ObjectIdentifier object_identifier: Controls yolo processing
        :param Drawer dawer: static opencv-based frame drawer
        :param dict settings: settings
        '''
        self.name = name
        self.feed_type = feed_type
        x = input("type an integer for camera id (Should be '0' or '1'. If '0' doesn't work, '1' should): ")
        x = int(x)
        self.source = x
        if feed_type == 'webrtc':
            # source is now a key + subkey pair to look up in process_bus
            self.source = (name, 'webrtc')
        self.process_bus = process_bus
        self.object_identifier = object_identifier
        self.drawer = drawer
        self.features = settings['features']
        self.thread_objects: List[threading.Thread] = list()
        self.register_with_video_manager: List[Tuple[str, str]] = list()
        self.video_manager: VideoManager = None
        self.register_with_data_sink: List[Tuple[str, str]] = list()
        self.data_sink_settings = settings['data_sink']
        self.data_sink: DataSink = None
        self.barrier: threading.Barrier = None
        self.feed_specific_setup()

    def __str__(self) -> str:
        """Object Identifier"""
        return self.name

    def run(self) -> None:
        """
        Main Feed Process execution block
        Execution ingests and digests a video frame and its analyzed state
        """
        processes = list()
        if self.barrier is None:
            raise NotImplementedError("barrier must be " +
                                      "defined in 'feed_specific_setup'")
        self.video_manager = VideoManager(process_bus=self.process_bus,
                                          source=self.source,
                                          barrier=self.barrier,
                                          input_type=self.feed_type,)
        self.data_sink = DataSink(name=str(self),
                                  process_bus=self.process_bus,
                                  settings=self.data_sink_settings)
        # this must come before starting the video_manager or data_sink
        # process in order to allow the queues to be registered prior
        # to them running. Threads deal with this oddly, so just be safe
        # and register the queues before starting the thread/process
        self.feed_specific_execution()
        threading.Thread(target=self.video_manager.run,
                         name=str(self.video_manager)).start()
        logging.info(f"FeedPipeline {str(self)} starting video_manager")
        data_sink_process = multiprocessing.Process(
            target=self.data_sink.run,
            name=str(self.data_sink))
        data_sink_process.start()
        logging.info(f"FeedPipeline {str(self)} starting data_sink")
        processes.append(data_sink_process.sentinel)
        logging.info(f"FeedPipeline {str(self)} DataSink Sentinel: " +
                     f"{data_sink_process.sentinel}")
        wait_list = multiprocessing.connection.wait(processes)
        while True:
            if wait_list:
                os.kill(0, 0)

    @abstractmethod
    def feed_specific_setup(self) -> None:
        raise NotImplementedError("Implement feed_specific_setup")

    @abstractmethod
    def feed_specific_execution(self) -> None:
        raise NotImplementedError("Implement feed_specific_execution")


class FaceRecognizerFeed(FeedPipeline):
    def __init__(self,
                 name: str,
                 feed_type: str,
                 source: Union[int, str],
                 process_bus: ProcessBus,
                 object_identifier: ObjectIdentifier,
                 drawer: Drawer,
                 settings: Dict[str, Any],):
        FeedPipeline.__init__(self,
                              name=name,
                              feed_type=feed_type,
                              source=source,
                              process_bus=process_bus,
                              object_identifier=object_identifier,
                              drawer=drawer,
                              settings=settings,)

    def feed_specific_setup(self) -> None:
        '''
        This is called in __init__(). It is necessary to be able to
        modify other member variables in classes running in other processes.
        For example:
            object_identifier has a self.round_robin member, which is a list of
            tuples for receiving a Frame and returning its results. If this
            list isn't appended to in the __init__() of a FeedPipeline, then
            the data will be lost.
        '''
        self.barrier = threading.Barrier(2)
        face_recognizer = FaceRecognizer(
                name=self.features['face_recognizer']['name'],
                process_bus=self.process_bus,
                barrier=self.barrier,
                drawer=self.drawer,
                settings=self.features['face_recognizer']['settings'])
        # self.object_identifier.register_customers(
        #         in_q_owner=str(self.object_identifier),
        #         in_q_name=str(self),
        #         out_q_owner=str(face_recognizer),
        #         out_q_name=f'in_q')
        # NOTE: These are necessary for process management related reasons.
        #   Meaning...becuase video_manager and data_sink are spawned in
        #   the main FeedPipeline.run for multiprocessing reasons, we need
        #   to do the registration within that function, which is why we need
        #   to save a reference to who needs to be registered and then use that
        #   in feed_specific_execution
        self.register_with_video_manager.append((str(face_recognizer),
                                                 'in_q'))
        self.register_with_data_sink.append((str(face_recognizer), 'out_q'))
        self.thread_objects.append(face_recognizer)

    def feed_specific_execution(self) -> None:
        '''
        This is called in the FeedPipeline.run (a new process). This is
        necessary to be able to acccess process-bound objects and control
        threads within the process
        '''
        for owner, q in self.register_with_video_manager:
            self.video_manager.register_out(owner, q)
        for owner, q in self.register_with_data_sink:
            self.data_sink.register_in(owner, q)
        for thread_object in self.thread_objects:
            logging.info(f"FeedPipeline {str(self)} starting {thread_object}")
            threading.Thread(target=thread_object.run,
                             name=str(thread_object)).start()


__all__ = 'FaceRecognizerFeed',

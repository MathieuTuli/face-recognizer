from datetime import datetime
from signal import signal, SIGINT, SIG_DFL
from typing import List, Union, Dict, Any
import logging

# import psycopg2
import numpy as np
import prctl
import cv2
import os

from .process_bus import ProcessBus
from .components import Sink, Frame, MonoQueue, UnlimitedQueue


class VideoWriter:
    '''
    Parent class for writing a video to file from np.ndarray frames
    '''
    def __init__(self,
                 name: str,
                 frame_type: str,
                 settings: Dict[str, Any],):
        self.name = f"{name}_{frame_type}"
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.current_fps = 0.0
        self.fps_sensitivity = 3
        self.current_file_time: str = None
        self.current_file_path: str = None
        self.video_writer: cv2.VideoWriter = None
        # TODO ugly, but sync-service needs to be adapted a bit to change
        self.output_path = os.path.join(settings['output_path'], 'output')
        # self.s3sync = S3Sync(settings)

    def config_file_paths(self,
                          timestamp: datetime,) -> None:
        '''
        Anytime fps changes or the hour changes or there is cause for a new
        file, this is called to configure the files are properly created
        '''
        current_minute_second = timestamp.strftime(u"%M-%S")
        current_hour = timestamp.strftime(u"%H")
        current_day = timestamp.strftime(u"%d")
        current_month = timestamp.strftime(u"%Y-%m")
        dir_path = os.path.expanduser(os.path.join(
                self.output_path,
                current_month,
                current_day,
                current_hour,))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(
                dir_path,
                f"{self.name}_{current_minute_second}.avi")
        # if self.current_file_path is not None:
        #     self.s3sync.notify(self.current_file_path)
        self.current_file_path = file_path

    def confirm_writer(self,
                       frame: np.ndarray,
                       timestamp: datetime,) -> None:
        '''
        This will confirm the video writer before writing. Basically,
        just decide if a new file is needed (change in fps say) and create
        it if so, otherwise do nothing
        '''
        fps = max(3, 1 / (datetime.now() - timestamp).total_seconds())
        logging.debug(f"FPS: {fps}")
        current_file_time = timestamp.strftime(u"%Y-%m-%d-%H")
        # TODO: An initial file is created containing only the first frame,
        #   need to fix the creation of this useless file
        if abs(fps - self.current_fps) > self.fps_sensitivity or \
                self.video_writer is None or \
                current_file_time != self.current_file_time:
            if self.video_writer is not None:
                self.video_writer.release()
            self.current_fps = fps
            h, w, c = frame.shape
            self.current_file_time = current_file_time
            self.config_file_paths(timestamp)
            self.video_writer = cv2.VideoWriter(
                self.current_file_path,
                self.fourcc,
                fps,
                (w, h))
            signal(SIGINT, SIG_DFL)

    def write(self,) -> None:
        raise NotImplementedError("Implement the write function")


class RawVideoWriter(VideoWriter):
    def __init__(self,
                 name: str,
                 settings: Dict[str, Any],):
        VideoWriter.__init__(self,
                             name=name,
                             frame_type='raw',
                             settings=settings)

    def write(self,
              frame: Frame) -> None:
        timestamp = frame.timestamp
        raw_frame = frame.frame
        self.confirm_writer(raw_frame, timestamp)
        self.video_writer.write(raw_frame)


class ProcVideoWriter(VideoWriter):
    def __init__(self,
                 name: str,
                 settings: Dict[str, Any],):
        VideoWriter.__init__(self,
                             name=name,
                             frame_type='proc',
                             settings=settings)
        self.move_window = True

    def write(self,
              frame: Frame,
              overlays: List[np.ndarray],) -> None:
        '''
        Note that overlays can't just be overlayed directly by doing
        np.add -> the rgb values will cause the overlay to look very odd
        Hence, you need to mask the areas that will be overlayed, then overlay
        '''
        timestamp = frame.timestamp
        proc_frame = frame.frame
        for overlay in overlays:
            mask = np.equal(overlay, 0).astype('uint8')
            proc_frame = proc_frame * mask
            proc_frame = np.add(proc_frame, overlay)
        self.confirm_writer(proc_frame, timestamp)
        cv2.imshow("image", proc_frame)
        if self.move_window:
            cv2.moveWindow("image", 20, 20)
            self.move_window = False
        cv2.waitKey(1)
        self.video_writer.write(proc_frame)


class DataSink(Sink):
    def __init__(self,
                 name: str,
                 process_bus: ProcessBus,
                 settings: Dict[str, Any],):
        self.name = name
        self.input_qs: List[Union[MonoQueue, UnlimitedQueue]] = list()
        self.process_bus = process_bus
        self.raw_video_writer = RawVideoWriter(name, settings)
        self.proc_video_writer = ProcVideoWriter(name, settings)
        self.start_time: datetime = datetime.now()

    def __str__(self) -> str:
        return self.name

    def register_in(self,
                    owner: str,
                    q_name: str,) -> None:
        self.input_qs.append(self.process_bus.get_queue(owner, q_name))

    def run(self) -> None:
        '''
        grabbing from queues must be blocking. Features all receive the
          exact same frame from the VideoManager by use of a barrier,
          meaning the frame only passes through when all the features are
          ready to receive a new frame. Thus, if we block-get from the
          registered queues, we will receive the proper frame/overlays
        '''
        prctl.set_name(str(self))
        while True:
            overlays = list()
            feature_results = list()
            frame = None
            time = datetime.now()
            for q in self.input_qs:
                feature_result = q.get()
                # signalled that nothing came through so ignore it
                if feature_result.frame is None:
                    continue
                # all features hold a reference to the same frame, so just
                #   grab one
                # TODO: make this self evident
                if frame is None:
                    frame = feature_result.frame
                if frame.timestamp != feature_result.frame.timestamp:
                    raise ValueError(
                        "Frame timestamps -> The features are not synced")
                if feature_result.overlay is not None:
                    overlays.append(feature_result.overlay)
                feature_results.append(feature_result)
            if frame is None:
                continue
            self.raw_video_writer.write(frame)
            self.proc_video_writer.write(frame, overlays)
            print(f"FPS {(1 / (datetime.now() - time).total_seconds())}")

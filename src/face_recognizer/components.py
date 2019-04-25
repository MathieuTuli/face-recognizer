'''Components Help'''
from multiprocessing.queues import Queue
from multiprocessing import Lock
from dataclasses import dataclass
from contextlib import suppress
from threading import Barrier
from datetime import datetime
from typing import TypeVar, NamedTuple, NewType, List, Union, Any
from queue import Empty, Full
from abc import abstractmethod
import multiprocessing

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import numpy as np


QT = TypeVar('QT')


# TODO With Supression may be faster (than a lock say), but at what cost?
#   what other errors could arise that the suppression is causing to ignore?
class MonoQueue(Queue):
    def __init__(self):
        Queue.__init__(self,
                       maxsize=1,
                       ctx=multiprocessing.get_context("spawn"))
        self.lock = Lock()

    def put(self, contents: QT, block=True, timeout=None):
        self.lock.acquire()
        while not self.empty():
            Queue.get(self, block=False)
        # NOTE/TODO: this, because multiprocessing Queues are stupid, is
        # necessary. Explained in short, if you try to q.put_nowait() too
        # quickly, it breaks. For example, say you were in ipython,
        # and you typed to following
        # - q = MonoQueue()
        # - q.put_nowait(2)
        # - q.put_nowait(3)
        # - q.put_nowait(4)
        # - q.put_nowait(5)
        # EVEN THOUGH there is a Lock() to atomize the access to the Queue,
        # one of the non-first 'put_nowait()' calls will acquire the lock,
        # the 'self.empty()' call is apparently True, even though something is
        # actually in the queue, and then it will not '.get()' it and try to
        # put something in the queue, raise a 'Full' exception.
        # So basically, apparently if something tries to put in the queue too
        # quickly, everything breaks. And yes, I made a pytest to test this,
        # guess what, if you try to run a trace (debugger), aka you jus step
        # through, it works fine, but as soon as you just run it, it breaks.
        # UGH, maybe I'm dumb and am doing something wrong
        with suppress(Full):
            Queue.put(self, contents, block=block, timeout=timeout)
        self.lock.release()

    def put_nowait(self, contents: QT):
        self.put(contents, block=False)

    def get_nowait(self) -> QT:
        self.lock.acquire()
        contents = None
        with suppress(Empty):
            contents = Queue.get(self, block=False)
        self.lock.release()
        return contents

    def get(self, block=True, timeout=None) -> QT:
        return Queue.get(self, block=True, timeout=timeout)


# class MonoQueue:
#     q: Queue
#
#     def __init__(self) -> None:
#         self.q = multiprocessing.Queue(1)
#         self.lock = Lock()
#
#     def put(self, contents: QT) -> None:
#         self.lock.acquire()
#         while not self.q.empty():
#             self.q.get_nowait()
#         self.q.put(contents)
#         self.lock.release()
#
#     def put_nowait(self, contents: QT) -> None:
#         self.lock.acquire()
#         while not self.q.empty():
#             self.q.get_nowait()
#         self.q.put_nowait(contents)
#         self.lock.release()
#
#     def get_nowait(self) -> Optional[QT]:
#         self.lock.acquire()
#         contents = None
#         if not self.q.empty():
#             contents = self.q.get_nowait()
#         self.lock.release()
#         return contents
#
#     def get(self) -> Optional[QT]:
#         contents = self.q.get()
#         return contents
#
#     def empty(self) -> bool:
#         return self.q.empty()
#
#     def qsize(self) -> int:
#         return self.q.qsize()
#
#
# class UnlimitedQueue:
#     q: Queue
#
#     def __init__(self) -> None:
#         self.q = multiprocessing.Queue()
#
#     def put(self, contents: QT) -> None:
#         self.q.put(contents)
#
#     def put_nowait(self, contents: QT) -> None:
#         self.q.put_nowait(contents)
#
#     def get_nowait(self) -> Optional[QT]:
#         with suppress(Empty):
#             return self.q.get_nowait()
#
#     def get(self) -> Optional[QT]:
#         return self.q.get()
#
#     def empty(self) -> bool:
#         return self.q.empty()
#
#     def qsize(self) -> int:
#         return self.q.qsize()


class UnlimitedQueue(Queue):
    def __init__(self):
        Queue.__init__(self, ctx=multiprocessing.get_context('spawn'))

    def put(self, contents: QT, block=True, timeout=None):
        Queue.put(self, contents, block=block, timeout=timeout)

    def put_nowait(self, contents: QT):
        self.put(contents, block=False)

    def get_nowait(self) -> QT:
        contents = None
        with suppress(Empty):
            contents = Queue.get(self, block=False)
        return contents

    def get(self, timeout=None) -> QT:
        return Queue.get(self, block=True, timeout=timeout)


class Source:
    def __init__(self) -> None:
        self.out: List[Union[MonoQueue, UnlimitedQueue]]

    @abstractmethod
    def register_out(self,) -> None:
        raise NotImplementedError("Implemented register_out")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Implement run")


class Sink:
    def __init__(self) -> None:
        self.input_qs: List[UnlimitedQueue]

    @abstractmethod
    def register_in(self,) -> None:
        raise NotImplementedError("Implemented register_in")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Implement run")


class Transceiver:
    def __init__(self,
                 name: str,
                 # TODO import issue when process_bus.py import components.py
                 # but components.py imports process_bus.py
                 process_bus=None,
                 barrier: Barrier = None) -> None:
        self.name = name
        self.in_q: Union[MonoQueue, UnlimitedQueue] = process_bus.get_queue(
                owner=name,
                q_name='in_q') if process_bus else None
        self.out_q: Union[MonoQueue, UnlimitedQueue] = process_bus.get_queue(
                owner=name,
                q_name='out_q') if process_bus else None
        self.barrier = barrier
        self.process_bus = process_bus
        self.downstream: List[UnlimitedQueue] = list()

    def __str__(self) -> str:
        return f"{self.name}_Transceiver"

    @abstractmethod
    def register_out(self,) -> None:
        raise NotImplementedError("Implement register_out")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Implement run")

    def send_results(self, results: Any) -> None:
        """
        A Transceiver can also have many downstream clients waiting
        for the results, where out_q will go to a specific location, say
        data_sink for saving
        """
        self.out_q.put_nowait(results)
        for q in self.downstream:
            q.put_nowait(results)


class SharedTransceiver:
    def __init__(self, name: str) -> None:
        self.round_robin: List[MonoQueue] = list()
        self.name = name

    def __str___(self) -> None:
        return f"{self.name}_SharedTransceiver"

    @abstractmethod
    def register_customers(self,) -> None:
        raise NotImplementedError("Implement register_customers")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Implement run")


BGR = NewType('BGR', np.ndarray)


class Frame(NamedTuple):
    timestamp: datetime
    frame: BGR


class BBox(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


class TrackedObject(NamedTuple):
    object_type: str
    object_id: int
    detection_score: float
    bbox: BBox


class TrackerOutput(NamedTuple):
    tracked_objects: List[TrackedObject]
    disappeared_objects: List[int]
    frame: Frame
    overlay: np.ndarray


class CashState(NamedTuple):
    cash_number: int
    state: bool
    location: BBox


class OpenCashLineOutput(NamedTuple):
    cashes: List[CashState]
    frame: Frame
    overlay: np.ndarray


class ObjectIdentifierObject(NamedTuple):
    class_name: str
    score: float
    bbox: BBox


class ObjectIdentifierOutput(NamedTuple):
    objects: List[ObjectIdentifierObject]
    frame: Frame


class Overlay(NamedTuple):
    owner: str
    frame: BGR


class TimeInCashOutput(NamedTuple):
    frame: Frame
    estimated_wait_time: float
    previous_actual_wait_time: float
    overlay: None


class WeightedZone(NamedTuple):
    weight: float
    zone: Polygon


@dataclass
class TrackedPersonWaitTime:
    person_id: int
    centroid: Point
    first_seen: datetime
    last_seen: datetime
    current_zone: WeightedZone


class FaceLabel(NamedTuple):
    label: str
    score: float


class Face(NamedTuple):
    bbox: BBox
    labels: List[FaceLabel]


class FaceRecognizerOutput(NamedTuple):
    faces: List[Face]
    frame: Frame
    overlay: np.ndarray

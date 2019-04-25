from threading import Barrier
from typing import Any
from abc import abstractmethod

from .process_bus import ProcessBus
from .components import Transceiver
from .drawing import Drawer


class Feature(Transceiver):
    """
    Parent Feature class
    """
    def __init__(self,
                 name: str,
                 drawer: Drawer,
                 barrier: Barrier,
                 process_bus: ProcessBus,
                 draw: bool = False,):
        """
        Constructor
        """
        Transceiver.__init__(self, name, process_bus, barrier)
        self.name = name
        self.draw = draw
        self.drawer = drawer

    def run(self) -> None:
        """
        Must be overwritten
        """
        raise NotImplementedError("Must overwrite")

    @abstractmethod
    def process_frame(self,) -> Any:
        """
        Must be overwritten
        """
        raise NotImplementedError("Must overwrite")

    def __str__(self) -> str:
        """Object Identifier
        """
        raise NotImplementedError("Overwrite")

    def register_out(self,
                     q_owner: str,
                     q_name: str) -> None:
        """
        A feature will have its out_q, which feeds into data_sink, but it can
        also have dependencies, hence they need to receive the results as well,
        which is the purpose of downstream queues list
        """
        self.downstream.append(self.process_bus.get_queue(q_owner, q_name))


__all__ = 'Feature',

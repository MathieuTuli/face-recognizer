'''
ProcessBus Help
'''
from typing import Union, Dict

from .components import UnlimitedQueue, MonoQueue


class ProcessBus:
    ''' ProcessManager
    ProcessBus is a class with an element registered_queues, which is a
        dictionary of format:
    registered_queues = {owner1: {queue_name:queue1.1, queue_name: queue1.2,},
                        owner2:{queue_name: queue2},....}
    '''
    def __init__(self) -> None:
        '''Constructor
        '''
        self.registered_queues: Dict[str,
                                     Dict[str, Union[MonoQueue,
                                                     UnlimitedQueue]]] = dict()

    def get_queue(self,
                  owner: str,
                  q_name: str,) -> Union[MonoQueue, UnlimitedQueue]:
        '''
        Simple method to grab a queue by owner name and queue name
        '''
        if owner not in self.registered_queues:
            raise ValueError(f"{owner} not a registered owner")
        if q_name not in self.registered_queues[owner]:
            raise ValueError(f"{q_name} not a registered queue for {owner}")
        return self.registered_queues[owner][q_name]

    def register_queue(self,
                       owner: str,
                       queue_name: str,
                       queue: Union[MonoQueue, UnlimitedQueue]) -> None:
        '''Register yourself to the manager
        Queues are registered using the following structure:
        {**queue_owner_string**: {
            **queue_name_str**: multirpocessing.Queue()
            }
        }

        Note: **queue_name_str** is hardcoded and must be consistent across
                anyone who accesses it
        '''
        if not isinstance(queue, MonoQueue) and \
                not isinstance(queue, UnlimitedQueue):
            raise ValueError("queue must be of type 'MonoQueue' or " +
                             "'UnlimitedQueue'")
        if owner not in self.registered_queues:
            self.registered_queues[owner] = {
                    queue_name: queue
                    }
        else:
            self.registered_queues[owner][queue_name] = queue


__all__ = 'ProcessBus',

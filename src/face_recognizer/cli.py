import logging

from enum import Enum
from argparse import ArgumentParser


class LogLevel(Enum):
    '''
    What the stdlib did not provide!
    '''
    # TODO: Global config [level], per module config [level]
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self):
        return self.name


parser = ArgumentParser(description=__doc__)
parser.add_argument('--log-level',
                    choices=LogLevel.__members__.values(),
                    type=LogLevel.__getitem__)

subparsers = parser.add_subparsers()

# TODO: Copied from rpi-surveillance/bin/preview. Dunno if good
for mode in 'files', 'webcam', 'server':
    subparser = subparsers.add_parser(mode)
    subparser.set_defaults(role=mode)
    if mode == 'files':
        pass
    elif mode == 'webcam':
        pass
    elif mode == 'server':
        pass


__all__ (,)

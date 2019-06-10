import importlib.resources
import multiprocessing
import logging
import sys
import os

from typing import Union

import pathlib
import prctl
import raven
import yaml

from .object_identifier import ObjectIdentifier
from .feed_pipeline import FaceRecognizerFeed
from .process_bus import ProcessBus
from .components import UnlimitedQueue  # , MonoQueue
from .drawing import Drawer


logging.root.setLevel(logging.INFO)
try:
    from systemd.journal import JournalHandler
    import systemd.daemon
except ImportError:
    def notify_startup_done() -> None:
        pass
else:
    if not os.isatty(sys.stdin.fileno()):
        logging.root.addHandler(JournalHandler())

    def notify_startup_done() -> None:
        systemd.daemon.notify('READY=1')

sentry = raven.Client()  # installs sys.excepthook too
DEFAULT_CONFIG = importlib.resources.path('face_recognizer',
                                          'config.yaml')
DEFAULT_COCO_CLASSES = importlib.resources.path(
        'face_recognizer.model_data',
        'coco_classes.txt')


class System:
    def __init__(self,
                 config_path: Union[str, pathlib.Path],
                 coco_classes_path: Union[str, pathlib.Path]):
        '''Constructor
        :param config_path: configuration file (yaml) path
        :type config_path: str or pathlib.PosixPath
        :param coco_classes_path: path to the coco classes file
        :type coco_classes_path: str or pathlib.PosixPath
        '''
        settings = dict()
        config_path = pathlib.Path(config_path)
        with config_path.open() as yaml_file:
            settings = yaml.load(yaml_file)

        feed_processes_settings = settings['Feeds']
        obj_id_settings = settings['object_identifier']

        self.feed_processes = dict()
        process_bus = ProcessBus()
        # NOTE: You can't use monoqueues, otherwise, two independent threads
        #   could drop different frames and then your videos/overlays don't
        #   sync up
        # TODO: Right now queues are referenced within other classes explicitly
        #       that means the names need to be properly specified here, which
        #       is dangerous
        for feed_name, settings in feed_processes_settings.items():
            for feature, feature_settings in settings['features'].items():
                name = feature_settings['name']
                process_bus.register_queue(owner=f"{name}",
                                           queue_name="in_q",
                                           queue=UnlimitedQueue())
                process_bus.register_queue(owner=f"{name}",
                                           queue_name="out_q",
                                           queue=UnlimitedQueue())
                if feature_settings['depends_on'] == 'object_identifier':
                    process_bus.register_queue(owner=obj_id_settings['name'],
                                               queue_name=f"{feed_name}",
                                               queue=UnlimitedQueue())

        coco_classes_path = pathlib.Path(coco_classes_path)
        with coco_classes_path.open() as coco_classes_file:
            class_names = coco_classes_file.readlines()

        class_names = [c.strip() for c in class_names]
        drawer = Drawer(class_names)
        if obj_id_settings['feature_on']:
            print("STARTING OBJECT IDENTIFIER")
            object_identifier = ObjectIdentifier(process_bus=process_bus,
                                                 class_names=class_names,
                                                 settings=obj_id_settings,)
            self.obj_id_process = multiprocessing.Process(
                    target=object_identifier.run,
                    name=obj_id_settings['name'])
        else:
            object_identifier = None
            self.obj_id_process = None
        # self.feed_processes is a dictionary with details about all the feeds:
        # {feed_name1:{object: <Pipeline>, process: <multiprocessing.Process>}}
        feed_pipeline_types = {
            'FaceRecognizerFeed': FaceRecognizerFeed,
        }
        for feed_name, settings in feed_processes_settings.items():
            feed_pipeline_type = settings['feed_pipeline_type']
            if feed_pipeline_type not in feed_pipeline_types:
                raise ValueError("Unkonwn 'feed_pipeline_type' " +
                                 f"{feed_pipeline_type}")
            feed_pipeline_object = feed_pipeline_types[feed_pipeline_type](
                    name=feed_name,
                    feed_type=settings['feed_type'],
                    source=settings['source'],
                    process_bus=process_bus,
                    object_identifier=object_identifier,
                    drawer=drawer,
                    settings=settings,)
            feed_pipeline = multiprocessing.Process(
                target=feed_pipeline_object.run,
                name=feed_name)
            self.feed_processes[feed_name] = {
                'object': feed_pipeline_object,
                'process': feed_pipeline,
            }

    def run(self) -> None:
        """
        run system
        """
        processes = list()
        if self.obj_id_process is not None:
            self.obj_id_process.start()
            processes.append(self.obj_id_process.sentinel)
            logging.info(f"YOLO Sentinel: {self.obj_id_process.sentinel}")
        for feed_name, contents in self.feed_processes.items():
            contents['process'].start()
            processes.append(contents['process'].sentinel)
            logging.info(
                    f"{feed_name} Sentinel: {contents['process'].sentinel}")
        wait_list = multiprocessing.connection.wait(processes)
        while True:
            if wait_list:
                os.kill(0, 0)


__all__ = 'System',


if __name__ == '__main__':
    config_posixpath = next(DEFAULT_CONFIG.gen)
    coco_classes_posixpath = next(DEFAULT_COCO_CLASSES.gen)
    system = System(config_path=config_posixpath,
                    coco_classes_path=coco_classes_posixpath)
    notify_startup_done()
    logging.info('\n\nStarting main.')
    system.run()

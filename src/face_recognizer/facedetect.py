import collections
import multiprocessing
import tensorflow as tf
import pickle
import numpy as np
import os
import facenet
import glob
import math
import detect_face
import time

from tensorflow.python.platform import gfile
from skimage.transform import resize

from .facenet import prewhiten, crop, flip

class FaceDetect(multiprocessing.Process):
    def __init__(self):
        """

        """
        multiprocessing.Process.__init__(self)
        # self.idx = idx
        # self.inputQ = inputq
        # self.outputQ = outputq
        # self.stopEvent = stop_event
        # self.settings = settings
        # self.currentInputDict = {}
        # self.yolo_class_names = class_names
        # self.faceid_class = '{}'.format(target)

        # config = tf.ConfigProto(
        #     device_count={'GPU': 0}
        # )

        self.saver = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.embedding_size = None
        self.image_size = 160

        config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2),
            device_count = {'GPU':1}
        )
        self.mtcnn_graph = tf.Graph()
        self.mtcnn_sess = tf.Session(graph=self.mtcnn_graph,config=config)
        with self.mtcnn_graph.as_default():
            with self.mtcnn_sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.mtcnn_sess, None)
                self.min_size = 20
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor

    def detect_faces(self, frame):
        if frame.size:
            bounding_boxes, _ = detect_face.detect_face(np.array(frame), self.min_size, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            images = np.zeros((len(bounding_boxes), self.image_size, self.image_size, 3))
            for i, box in enumerate(bounding_boxes):
                images[i, :, :, :] = self.prepare_images(frame, box, self.image_size)

            return images, bounding_boxes

    def prepare_images(self, image, box, image_size, do_random_crop=False, do_random_flip=False, do_prewhiten=True):
        img = np.array(image.crop(box=(box[0], box[1], box[2], box[3])))
        img = resize(img, (image_size, image_size))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = facenet.crop(img, do_random_crop, image_size)
        img = facenet.flip(img, do_random_flip)
        return img

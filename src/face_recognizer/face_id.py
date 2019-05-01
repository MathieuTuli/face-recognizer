from tensorflow.python.platform import gfile
from skimage.transform import resize

import importlib.resources
import tensorflow as tf
import pickle
import numpy as np
import os
import sys

from .components import Face, FaceLabel, BBox
from .facenet import prewhiten
import facenet

FACE_ID_PB_FILE = next(importlib.resources.path(
        'face_recognizer.model_data',
        'classifier.pb').gen)


class FaceID:
    def __init__(self, idx, class_names, settings):
        """
        :param list idx: integer list to distinguish objects [type,sub-type]
        :param multiprocessing.Manager.Queue() inputq: each element is a
            dictionary with keys, 'original_image','frame_id',
                ['boxes'],['scores']['classes']
        boxes: list of [[y_min,x_min,y_max,x_max], [y_min,..],...] where
        | x_min, y_min |   |             |
        |              |   |             |
        |              |   | x_max,y_max |
        scores: list of probabilities for each detection
        'classes': list of classes for each detection
        :param multiprocessing.Manager.Queue() outputq: Each element is a
            dictionary with keys, 'idx'(=self.idx),
                'frame_id'(=inputq.get()['frame_id']), ['boxes'],['scores'],
                    ['classes'] for recognized face.
        (These have been defined as lists to allow the possibility of
            recognizing multiple faces from the same class)
        :param multiprocessing.Event() stop_event: Signals the process to stop
        :param dict target: {'target': 'item to track'},
            eg: {'target':'henri'}. henri refers to a
                faceid_models/*_classifier.pkl file
        """
        target = settings['target']
        model_name = settings['model_name']
        names = target.split('_')
        self.faceid_class = []
        for name in names:
            if name.find("not") != -1:
                continue
            self.faceid_class.append('{}'.format(name.strip()))

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.facenet_graph_def = tf.GraphDef()
        self.facenet_graph = tf.Graph()
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.embedding_size = None
        self.image_size = 160

        self.init_model(str(FACE_ID_PB_FILE), model_name)

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=settings['gpu_frac']),
            device_count={'GPU': 1}
        )
        self.facenet_sess = tf.Session(graph=self.facenet_graph, config=config)

        print('{} classifier running.'.format(target))

    def identify(self, images, bounding_boxes, threshold,
                 show_all_labels=False):
        feed_dict = {self.images_placeholder: images,
                     self.phase_train_placeholder: False}
        emb_array = np.zeros((len(images), self.embedding_size))

        emb_array[:, :] = self.facenet_sess.run(self.embeddings,
                                                feed_dict=feed_dict)

        predictions = self.model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[
                np.arange(len(best_class_indices)), best_class_indices]

        results = list()
        # don't use this just yet (or really ever)
        if show_all_labels:
            predictions = predictions.tolist()
            pass
        else:
            for i in range(len(best_class_indices)):
                # if best_class_probabilities[i] >= threshold:
                if self.class_names[best_class_indices[i]] \
                        in self.faceid_class:
                    indx = self.faceid_class.index(
                            self.class_names[best_class_indices[i]])
                    # return [(self.faceid_class[indx].encode('utf-8'),
                    #         best_class_probabilities[i], (x,y,w,h))]
                    results.append(Face(
                        bbox=BBox(
                            left=bounding_boxes[i][0],
                            top=bounding_boxes[i][1],
                            right=bounding_boxes[i][2],
                            bottom=bounding_boxes[i][3]),
                        labels=[FaceLabel(
                            label=self.faceid_class[indx],
                            score=best_class_probabilities[i])]))
        return results

    def init_model(self, model, model_name, input_map=None):
        # Check if the model is a model directory (containing a metagraph and
        # a checkpoint file) or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        with self.facenet_graph.as_default():
            if (os.path.isfile(model_exp)):
                print('Model filename: %s' % model_exp)
                with gfile.FastGFile(model_exp, 'rb') as f:
                    self.facenet_graph_def = tf.GraphDef()
                    self.facenet_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(self.facenet_graph_def,
                                        input_map=input_map, name='')
            else:
                print("Oops. Couldn't find the classifier.pb")
                sys.exit(0)
                # print('Model directory: %s' % model_exp)
                # meta_file, ckpt_file = get_model_filenames(model_exp)
                #
                # print('Metagraph file: %s' % meta_file)
                # print('Checkpoint file: %s' % ckpt_file)
                #
                # self.saver = tf.train.import_meta_graph(
                #     os.path.join(model_exp, meta_file), input_map=input_map)
                # self.saver.restore(self.graph, os.path.join(model_exp,
                #                                             ckpt_file))
            with open(model_name, 'rb') \
                    as infile:
                (self.model, self.class_names) = pickle.load(infile)
            print("Faceid class names: {}".format(self.class_names))

        self.images_placeholder = self.facenet_graph.get_tensor_by_name(
                "input:0")
        self.embeddings = self.facenet_graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.facenet_graph.get_tensor_by_name(
                "phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def prepare_images(self, image, box, image_size,
                       do_random_crop=False, do_random_flip=False,
                       do_prewhiten=True):
        img = np.array(image.crop(box=(box[0], box[1], box[2], box[3])))
        img = resize(img, (self.image_size, self.image_size))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = facenet.crop(img, do_random_crop, image_size)
        img = facenet.flip(img, do_random_flip)
        return img


__all__ = 'FaceID'

import tensorflow as tf
import numpy as np
import facenet

from PIL import Image
from facenet import prewhiten, crop, flip
from tensorflow.python.platform import gfile
from skimage.transform import resize

class FaceDetector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.image_size = 160
        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def detect(self, image, score_threshold=0.5):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        h, w, _ = image.shape
        pil_image = Image.fromarray(image)
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        bounding_boxes = []
        for box in boxes:
            bounding_boxes.append([box[1], box[0], box[3], box[2]])

        images = np.zeros((len(bounding_boxes), self.image_size, self.image_size, 3))
        for i, box in enumerate(bounding_boxes):
            images[i, :, :, :] = self.prepare_images(pil_image, box, self.image_size)

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
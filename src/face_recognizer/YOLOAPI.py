'''
YOLOAPI for YOLO3-4-Py
'''
import cv2
import os
import time
import colorsys
import json
import tkinter as tk

from blessings import Terminal
from pydarknet import Detector, Image
from datetime import datetime


class YOLOAPI():
    '''
    Class for API access to YOLO (YOLO3-4-Py wrapper)
    '''
    def __init__(self,
                 source=None,
                 is_image=False,
                 threshold=0.7,
                 preview=False,
                 use_slider=False,
                 save_results=False,
                 save_video=False,
                 class_names_file="coco_classes.txt",
                 class_filters=None):
        '''
        @param source: int or str | video source
            default=None
        @param is_image: boolean | whether source is image or not
        @param threshold: float | detection threshold
            default=0.7
        @param preview: boolean | whether or not to display cv2 window
            default=False
        @param use_slider: boolean | whether to create slider for thresholding
            default=False
        @param save_results: boolean | whether to save json results to file
            default=False
        @param save_video: boolean | whether to save video to file
            default=False
        @param class_names_file: str | file for class names
            default="coco_classes.txt"
        '''
        self.threshold = threshold

        self.use_slider = use_slider
        if self.use_slider:
            self.gui = tk.Tk()
            self.slider = tk.Scale(self.gui,
                                   from_=0.05,
                                   to=0.95,
                                   resolution=0.05,
                                   orient=tk.HORIZONTAL)
            self.slider.set(0.7)
            self.slider.pack()

        self.class_filters = class_filters
        if class_filters is not None:
            assert type(class_filters) == list, "@param 'class_filters' must be a list"
            self.class_filters = class_filters
        self.class_names = list()
        self.colors = list()
        self.init_names_colors(class_names_file)

        self.net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"),
                            bytes("weights/yolov3.weights", encoding="utf-8"),
                            0,
                            bytes("cfg/coco.data", encoding="utf-8"))

        if source is not None:
            if is_image:
                self.cap = cv2.imread(source)
            else:
                self.cap = cv2.VideoCapture(source)

            self.preview = preview
            if preview:
                cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("preview", 1920, 1080)
                self.term = Terminal()

            self.frame_dims = [self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                               self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]

            self.results_file = None
            if save_results:
                self.datetime = datetime.now().isoformat()

                self.results_file = os.path.join(
                                        "results_files",
                                        "{}_results.json".format(
                                            self.datetime))
                self.total_results = dict()
                self.total_results["frame_IDs"] = list()

            self.video_file_out = None
            if save_video:
                self.datetime = datetime.now().isoformat()
                self.video_file_out = os.path.join("video_files_out",
                                                   "{}_video.mp4".format(
                                                        self.datetime))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.video_file_out,
                                                    fourcc,
                                                    60.0,
                                                    (int(self.frame_dims[0]),
                                                     int(self.frame_dims[1])))
                print("Ready to write video")

    def init_names_colors(self, class_names_file):
        '''
        Initialize class names and colors for bounding boxes

        @param class_names_file: str | location of class names *.txt file
        '''
        # colors and class names
        with open(class_names_file) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        # Colors are (b,g,r)
        length = len(self.class_names)
        for i in range(length):
            hsv = (i / length, 1., 1.)
            rgb = colorsys.hsv_to_rgb(*hsv)
            self.colors.append((int(rgb[0] * 255),
                                int(rgb[1] * 255),
                                int(rgb[2] * 255)))

    def run(self):
        '''
        Run YOLO, system defined in constructor
        '''
        frame_ID = 0
        beginning_time = time.time()
        while True:
            r, frame = self.cap.read()
            if not r:
                self.close()
                break

            start_time = time.time()

            if self.use_slider:
                self.gui.update_idletasks()
                self.gui.update()
                self.threshold = self.slider.get()

            results, frame = self.get_detections(frame, self.threshold)
            frame = self.draw(results, frame)

            if self.preview:
                to_print = [
                    "FPS: {:.2f}".format(1 / float(time.time() - start_time)),
                    "Time: {:.2f}".format(time.time() - beginning_time),
                    "Frames: {}".format(frame_ID)
                ]
                os.system('clear')
                for i, prints in enumerate(to_print):
                    with self.term.location(y=i):
                        print(prints)

                cv2.imshow("preview", frame)
                key = cv2.waitKey(1)
                if key == 0xFF & ord("q"):
                    self.close()
                    break

            if self.results_file is not None:
                self.save_results(self.results_file, frame_ID, results)

            if self.video_file_out is not None:
                self.save_video(self.video_writer, frame)

            frame_ID += 1

    def draw(self, results, frame=None, draw_scores=True):
        '''
        Draw bounding boxes

        @param results: list
            *note* -> each element in the list must be a tuple of
                (class, score, bounding_box)
                   -> class is a str 'utf-8' encoded
                   -> score is a float
                   -> bounding_box is (x,y,w,h), where (x,y) is the centroid
        @param frame: numpy.ndarray
        @param draw_scores: boolean

        @return frame: np.ndarray
        '''
        if frame is None:
            return
        for cat, score, bounds in results:
            x, y, w, h = bounds
            i = int(self.class_names.index(str(cat.decode("utf-8"))))

            padding = 5
            font_size = 0.7
            font_thickness = 1

            if draw_scores:
                text = str(cat.decode("utf-8")) + " {:.2f}".format(score)
            else:
                text = str(cat.decode("utf-8"))

            size = cv2.getTextSize(text,
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=font_size,
                                   thickness=font_thickness)

            cv2.rectangle(img=frame,
                          pt1=(int(x - w / 2),
                               int(y - h / 2) - size[0][1] - 3 * padding),
                          pt2=(int(x - w / 2) + size[0][0] + 4 * padding,
                               int(y - h / 2)),
                          color=self.colors[i],
                          thickness=-1)
            cv2.putText(img=frame,
                        text=text,
                        org=(int(x - (w / 2)) + 2 * padding, int(y - (h / 2)) - 2 * padding),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_size,
                        color=(255, 255, 255),
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA)
            cv2.rectangle(img=frame,
                          pt1=(int(x - w / 2), int(y - h / 2)),
                          pt2=(int(x + w / 2), int(y + h / 2)),
                          color=self.colors[i],
                          thickness=1)
        return frame

    def get_detections(self, frame, threshold):
        '''
        Get YOLO detections

        @param frame: np.ndarray
        @param threshold: float

        @return results: list | of tuples (class, score, bounding_box)
            bounding_box: (x,y,w,h) where (x,y) is the centroid
        @return frame: np.ndarray
        '''
        dark_frame = Image(frame)
        results = self.net.detect(dark_frame, self.threshold)
        del dark_frame
        if self.class_filters is not None:
            results = [(cat, score, bounds) for cat, score, bounds in results if cat.decode('utf-8') in self.class_filters]
        return results, frame

    def save_results(self, file_name, frame_ID, results):
        '''
        Save results to a file

        @param file_name: str | file name to save to
        @param frame_ID: <anything> | user define frame_ID to reference in json
        @param results: list | results from get_detections
        '''
        frame_objects = list()
        for cat, score, bounds in results:
            x, y, w, h = bounds
            label = str(cat.decode("utf-8"))
            score = "{:.2f}".format(score)
            frame_objects.append({
                "label": label,
                "score": score,
                "bbx [x,y,w,h]": [x, y, w, h],
                "frame_dims [w,h]": self.frame_dims
            })

        results_dict = {frame_ID: frame_objects}
        self.total_results["frame_IDs"].append(results_dict)

        with open(file_name, "w+") as f:
            json.dump(self.total_results, f)

    def save_video(self, video_writer, frame):
        '''
        Save processed video to file

        @param frame: np.ndarray
        @param video_writer: cv2.VideoWriter()
        '''
        if video_writer is not None:
            try:
                video_writer.write(frame)
            except:
                print("Couldn't write video")

    def close(self):
        '''
        Release self.cap and self.video_writer if defined
        '''
        if self.cap is not None:
            self.cap.release()
        if self.video_file_out is not None:
            self.video_writer.release()

if __name__ == "__main__":
    # yolo = YOLOAPI(source=0,
    #                threshold=0.5,
    #                preview=True,
    #                use_slider=True,
    #                save_results=False,
    #                save_video=False)
    # yolo.run()
    yolo = YOLOAPI(source='BF.mp4',
                   is_image=False,
                   threshold=0.7,
                   preview=True,
                   use_slider=False,
                   save_results=False,
                   save_video=False,
                   class_filters=['person'])
    yolo.run()


import cv2
import argparse
import os
import random

from datetime import datetime
from YOLOAPI import YOLOAPI
from facedetect_v2 import FaceDetector
from faceid import FaceID

parser = argparse.ArgumentParser()
parser.add_argument("--cam",
                    dest="cam",
                    default=0,
                    type=int)
args = parser.parse_args()


class YOLOFace():
    def __init__(self, cam, to_track: list, save_video=False,
                 detect_face=False, show_scores=True, show_all=False,
                 model_filename=None):
        '''
        @param cam: int | camera index
        @param to_track: list | people names seperated by '_'
        @param save_video: boolean
        @param detect_face: boolean
        '''
        if model_filename is None:
            raise ValueError('wrong model_filename')
        self.yolo = YOLOAPI()
        self.yolo.class_names.append('face')
        self.yolo.colors.append((210, 170, 0))
        self.face_detect = FaceDetector('facenet/faceid_model/detect.pb')
        self.save_video = save_video
        self.detect_face = detect_face
        self.show_scores = show_scores
        self.show_all = show_all

        for who in to_track.split('_'):
            self.yolo.class_names.append(who)
            self.yolo.colors.append((210, 170, 0))
        self.color = (210, 170, 0)
        self.face_id = FaceID([1, 1],
                              self.yolo.class_names,
                              None,
                              None,
                              None,
                              {'target': to_track,
                               'gpu_frac': 0.1,
                               'model_name': '{}'.format(model_filename)})

        self.cam = cam
        self.size = None
        self.once_draw = True

    def run(self):
        cap = cv2.VideoCapture(self.cam)
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", 1920, 1080)

        if self.save_video:
            frame_dims = [cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
            dtime = datetime.now().isoformat()
            video_file_out = os.path.join("video_files_out",
                                          "{}_video.mp4".format(dtime))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_file_out,
                                           fourcc,
                                           24.0,
                                           (int(frame_dims[0]),
                                            int(frame_dims[1])))

        while True:
            r, frame = cap.read()
            # frame = cv2.imread('/home/imrsv/Downloads/wayne.jpg')
            if not r:
                cap.release()
                break

            results, frame = self.yolo.get_detections(frame, threshold=0.7)
            frame = self.yolo.draw(results, frame, self.show_scores)

            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, boxes = self.face_detect.detect(image_array)

            if self.detect_face:
                results = []
                for box in boxes:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    x = box[0] + w/2
                    y = box[1] + h/2
                    results.append(('face'.encode('utf-8'),
                                   random.uniform(0.8, 0.89), (x, y, w, h)))
                frame = self.yolo.draw(results, frame, self.show_scores)
            else:
                if faces.any():
                    results = self.face_id.identify(faces, boxes, threshold=0.65, show_all=self.show_all)
                    if results:
                        if self.show_all:
                            frame = self.draw_faces(results, frame)
                        else:
                            frame = self.yolo.draw(results, frame, self.show_scores)

            if self.save_video:
                video_writer.write(frame)

            cv2.imshow("preview", frame)
            key = cv2.waitKey(1)
            if key == 0xFF & ord("q"):
                break

    def draw_faces(self, results, frame):
        if frame is None:
            return

        faceid_classes, class_names, stats = results
        for scores, bounds in stats:
            x, y, w, h = bounds

            font_size = 0.7
            text_padding = 7
            padding = int(text_padding * (len(faceid_classes) / 2)) + 4
            font_thickness = 1

            if self.once_draw:
                text = faceid_classes[0] + " {:.2f}".format(scores[0])
                max_size = cv2.getTextSize(text,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=font_size,
                                           thickness=font_thickness)
                for t in faceid_classes:
                    text = t + " {:.2f}".format(scores[0])
                    size = cv2.getTextSize(text,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=font_size,
                                           thickness=font_thickness)
                    if size[0][0] > max_size[0][0]:
                        max_size = size
                self.size = max_size
                self.once_draw = False

            cv2.rectangle(img=frame,
                          pt1=(int(x - w / 2),
                               int(y - h / 2) - self.size[0][1] * len(scores) - 2 * padding),
                          pt2=(int(x - w / 2) + self.size[0][0],
                               int(y - h / 2)),
                          color=self.color,
                          thickness=-1)
            for i, score in enumerate(scores):
                text = faceid_classes[faceid_classes.index(class_names[i])] + " {:.2f}".format(score)

                cv2.putText(img=frame,
                            text=text,
                            org=(int(x - (w / 2)), int(y - (h / 2)) - (self.size[0][1] * i) - (text_padding * (i + 1))),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_size,
                            lineType=cv2.LINE_AA,
                            thickness=font_thickness,
                            color=(255, 255, 255))

            cv2.rectangle(img=frame,
                          pt1=(int(x - w / 2), int(y - h / 2)),
                          pt2=(int(x + w / 2), int(y + h / 2)),
                          color=self.color,
                          thickness=1)
        return frame

if __name__ == "__main__":
    print("Camera: {}".format(args.cam))
    yoloFace = YOLOFace(cam=args.cam,
                        to_track='Samuel Witherspoon_Brad Pitt_Angelina Jolie',
                        save_video=False,
                        detect_face=False,
                        show_scores=True,
                        show_all=True,
                        model_filename='brad_sam_angelina')
    yoloFace.run()


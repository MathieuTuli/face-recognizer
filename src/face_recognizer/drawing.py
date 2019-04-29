"""
Drawer help
"""

from typing import List, Tuple
import cv2
import colorsys
import numpy as np

from .components import CashState, TrackedObject, Face


class Drawer:
    """Drawing for modules"""
    def __init__(self, class_names: List[str]) -> None:
        """Constructor
        @param class_name: list
            @list-element: string
        """
        self.colors: List[Tuple[int, int, int]] = list()  # colors are (b,g,r)
        self.class_names = class_names
        length = len(class_names)
        for i in range(length):
            hsv = (i / length, 1., 1.)
            rgb = colorsys.hsv_to_rgb(*hsv)
            self.colors.append((int(rgb[0] * 255),
                                int(rgb[1] * 255),
                                int(rgb[2] * 255)))

        # custom coloring
        self.colors[class_names.index("person")] = (60, 4, 204)
        self.cash_on_color = (0, 255, 0)
        self.cash_off_color = (0, 0, 255)
        self.face_colour = (255, 0, 0)

    def __str__(self) -> str:
        '''
        '''
        return 'Drawer'

    def draw_faces(self,
                   frame: np.ndarray,
                   results: List[Face]) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        # Drawing parameters
        font_size = (1.5 * frame_width) / 1664
        score_font_size = 0.6 * font_size
        thickness = int((2 * frame_width) / 1664)
        thickness = 1 if thickness == 0 else thickness
        padding = int((15 * frame_width) / 1664)
        line_type = cv2.LINE_AA
        font_type = cv2.FONT_HERSHEY_SIMPLEX

        for result in results:
            bbox = result.bbox
            x1 = bbox.left
            y1 = bbox.top
            x2 = bbox.right
            y2 = bbox.bottom
            bbx_x1 = x1 if x1 > 0 else 1
            bbx_y1 = y1 if y1 > 0 else 1
            bbx_x2 = x2 if x2 < frame_width else frame_width - 1
            bbx_y2 = y2 if y2 < frame_height else frame_height - 1

            cv2.rectangle(img=frame,
                          pt1=(bbx_x1, bbx_y1),
                          pt2=(bbx_x2, bbx_y2),
                          color=self.face_colour,
                          thickness=thickness)
            face_labels = result.labels
            max_size = 0
            for face_label in face_labels:
                id_ = face_label.label
                score = face_label.score
                id_size = cv2.getTextSize(id_.lower(),
                                          fontFace=font_type,
                                          fontScale=font_size,
                                          thickness=thickness)
                if id_size > max_size:
                    max_size = id_size
            score_size = cv2.getTextSize(score.lower(),
                                         fontFace=font_type,
                                         fontScale=score_font_size,
                                         thickness=thickness)
            padding_text_left = 2 * padding
            # ratio = (x2-x1) / class_size[0][0]
            ratio = (x2 - x1) / (max_size[0][0] + padding_text_left)
            background_extender = 0
            if ratio < 1.0:
                background_extender = (max_size[0][0] + padding_text_left) -\
                        (x2 - x1) + padding

            top_text_background = y1 - int(max_size[0][1] +
                                           score_size[0][1]) * len(face_labels)
            box_top = top_text_background - 2 * padding
            frame_height_offset = 0
            if box_top < 0:
                frame_height_offset = abs(box_top)

            cv2.rectangle(img=frame,
                          pt1=(x1, box_top + frame_height_offset),
                          pt2=(x2 + background_extender,
                               y1 + frame_height_offset),
                          color=self.face_colour,
                          thickness=-1)
            for i, face_label in enumerate(face_labels):
                id_ = face_label.label
                score = face_label.score
                # text
                cv2.putText(img=frame,
                            text=id_,
                            org=(x1 + padding_text_left,
                                 int(top_text_background + padding +
                                     frame_height_offset +
                                     (i * max_size[0][1]))),
                            fontFace=font_type,
                            fontScale=font_size,
                            color=(255, 255, 255),
                            thickness=thickness,
                            lineType=line_type)
                cv2.putText(img=frame,
                            text=score,
                            org=(x1 + padding_text_left,
                                 int(y1 - padding + frame_height_offset +
                                     max_size[0][1] + i * max_size[0][1])),
                            fontFace=font_type,
                            fontScale=font_size,
                            color=(255, 255, 255),
                            thickness=thickness,
                            lineType=line_type)
        return frame

    def draw_tracker(self,
                     frame: np.ndarray,
                     results: List[TrackedObject]) -> np.ndarray:
        """
        Draw bounding boxes
        @param results: list
            @list-element: tuple
                @tuple-element class: str
                @tuple-element id: int
                @tuple-element score: float
                @tuple-element bounds: tuple
                    @tuple-element x: center x
                    @tuple-element y: center y
                    @tuple-element w: width
                    @tuple-element h: height
        @param frame: numpy.ndarray
        @return frame: numpy.ndarray
        """
        # bounds = yolo_results['boxes']
        # scores = yolo_results['scores']
        # classes = yolo_results['classes']
        # assert len(bounds) == len(scores) and len(scores) == len(classes)
        for result in results:
            class_ = result.object_type
            id_ = result.object_id
            bounds = result.bbox
            x1 = bounds.left
            y1 = bounds.top
            x2 = bounds.right
            y2 = bounds.bottom
            # x, y, w, h = bounds
            ind = self.class_names.index(class_)
            frame_height, frame_width = frame.shape[:2]
            # Drawing parameters
            font_size = (1.5 * frame_width) / 1664
            id_font_size = 0.6 * font_size
            thickness = int((2 * frame_width) / 1664)
            thickness = 1 if thickness == 0 else thickness
            padding = int((15 * frame_width) / 1664)
            line_type = cv2.LINE_AA
            font_type = cv2.FONT_HERSHEY_SIMPLEX
            # bounding box
            # x1 = int(x - (w/2))
            # x2 = int(x + (w/2))
            # y1 = int(y - (w/2))
            # y2 = int(y + (w/2))
            bbx_x1 = x1 if x1 > 0 else 1
            bbx_y1 = y1 if y1 > 0 else 1
            bbx_x2 = x2 if x2 < frame_width else frame_width - 1
            bbx_y2 = y2 if y2 < frame_height else frame_height - 1

            cv2.rectangle(img=frame,
                          pt1=(bbx_x1, bbx_y1),
                          pt2=(bbx_x2, bbx_y2),
                          color=self.colors[ind],
                          thickness=thickness)
            # text background
            class_ = class_.lower()
            id_ = "ID: {}".format(id_)
            class_size = cv2.getTextSize(class_.lower(),
                                         fontFace=font_type,
                                         fontScale=font_size,
                                         thickness=thickness)
            id_size = cv2.getTextSize(id_,
                                      fontFace=font_type,
                                      fontScale=font_size,
                                      thickness=thickness)
            padding_text_left = 2 * padding
            # ratio = (x2-x1) / class_size[0][0]
            ratio = (x2 - x1) / (class_size[0][0] + padding_text_left)
            background_extender = 0
            if ratio < 1.0:
                background_extender = (class_size[0][0] + padding_text_left) -\
                        (x2 - x1) + padding
                # font_size *= ratio
                # id_font_size *= ratio
                # class_size = cv2.getTextSize(class_.lower(),
                #                              fontFace=font_type,
                #                              fontScale=font_size,
                #                              thickness=font_thickness)
                # id_size = cv2.getTextSize(id_,
                #                           fontFace=font_type,
                #                           fontScale=font_size,
                #                           thickness=font_thickness)
            # padding_text_left = int(((x2-x1) - class_size[0][0]) / 2)

            top_text_background = y1 - int(class_size[0][1] + id_size[0][1])
            box_top = top_text_background - 2 * padding
            frame_height_offset = 0
            if box_top < 0:
                frame_height_offset = abs(box_top)

            cv2.rectangle(img=frame,
                          pt1=(x1, box_top + frame_height_offset),
                          pt2=(x2 + background_extender,
                               y1 + frame_height_offset),
                          color=self.colors[ind],
                          thickness=-1)
            # text
            cv2.putText(img=frame,
                        text=class_,
                        org=(x1 + padding_text_left,
                             int(top_text_background + padding +
                                 frame_height_offset)),
                        fontFace=font_type,
                        fontScale=font_size,
                        color=(255, 255, 255),
                        thickness=thickness,
                        lineType=line_type)
            cv2.putText(img=frame,
                        text=id_,
                        org=(x1 + padding_text_left,
                             int(y1 - padding + frame_height_offset)),
                        fontFace=font_type,
                        fontScale=id_font_size,
                        color=(255, 255, 255),
                        thickness=thickness,
                        lineType=line_type)
        return frame

    def draw_open_cash_line_state(self,
                                  frame: np.ndarray,
                                  results: List[CashState]) -> np.ndarray:
        """Draw [red/green] bounding box around cash register sign
            based, determining if it's [off/on]
        """
        for cash_state in results:
            is_on = cash_state.state
            location = cash_state.location
            left_top = (location.left, location.top)
            right_bottom = (location.right, location.bottom)
            color = self.cash_on_color if is_on else self.cash_off_color
            cv2.rectangle(img=frame,
                          pt1=left_top,
                          pt2=right_bottom,
                          color=color,
                          thickness=2)
        return frame


__all__ = 'Drawer',

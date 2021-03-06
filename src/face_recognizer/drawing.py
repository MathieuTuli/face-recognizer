"""
Drawer help
"""

from typing import List, Tuple
import colorsys
import numpy as np
import cv2

from .components import Face, ObjectIdentifierObject


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
        self.face_colour = (255, 255, 255)

    def __str__(self) -> str:
        '''
        '''
        return 'Drawer'

    def draw_faces(self,
                   frame: np.ndarray,
                   results: List[Face],
                   overlay: bool) -> np.ndarray:
        frame = frame * 0 if overlay else frame
        frame_height, frame_width = frame.shape[:2]
        # Drawing parameters
        font_size = (1.5 * frame_width) / 1664
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
            for i, face_label in enumerate(face_labels):
                id_ = face_label.label
                score = face_label.score
                score = round(score, 2)
                id_ += f" {str(score)}%"
                id_size = cv2.getTextSize(id_.lower(),
                                          fontFace=font_type,
                                          fontScale=font_size,
                                          thickness=thickness)
                if i == 0:
                    max_size = id_size
                else:
                    if id_size > max_size:
                        max_size = id_size
            # score_size = cv2.getTextSize(str(score),
            #                              fontFace=font_type,
            #                              fontScale=score_font_size,
            #                              thickness=thickness)
            padding_text_left = padding
            # ratio = (x2-x1) / class_size[0][0]
            ratio = (x2 - x1) / (max_size[0][0] + padding_text_left)
            background_extender = 0
            if ratio < 1.0:
                background_extender = (max_size[0][0] + padding_text_left) -\
                        (x2 - x1) + padding

            # top_text_background = y1 - (int(max_size[0][1] +
            #                                 score_size[0][1]) *
            #                             (len(face_labels))) - (2 * padding)
            top_text_background = y1 - (int(max_size[0][1]) *
                                        (len(face_labels)))
            box_top = top_text_background - 2 * padding
            frame_height_offset = 0
            if box_top < 0:
                frame_height_offset = abs(box_top)

            cv2.rectangle(img=frame,
                          pt1=(int(x1), int(box_top + frame_height_offset)),
                          pt2=(int(x2 + background_extender),
                               int(y1 + frame_height_offset)),
                          color=self.face_colour,
                          thickness=-1)
            multilabel_padding = 5 if len(face_labels) != 0 else 0
            for i, face_label in enumerate(face_labels):
                id_ = face_label.label
                score = round(face_label.score, 2)
                id_ += f" {str(score)}%"
                # text
                cv2.putText(img=frame,
                            text=id_,
                            org=(int(x1 + padding_text_left),
                                 int(top_text_background + padding +
                                     frame_height_offset +
                                     (i * max_size[0][1]) +
                                     multilabel_padding * i)),
                            fontFace=font_type,
                            fontScale=font_size,
                            color=(0, 0, 0),
                            thickness=thickness,
                            lineType=line_type)
                # cv2.putText(img=frame,
                #             text=str(score),
                #             org=(int(x1 + padding_text_left),
                #                  int(top_text_background + padding +
                #                      frame_height_offset +
                #                      max_size[0][1] + padding +
                #                      (i * max_size[0][1]))),
                #             fontFace=font_type,
                #             fontScale=font_size,
                #             color=(255, 255, 255),
                #             thickness=thickness,
                #             lineType=line_type)
        return frame

    def draw_bboxes(self,
                    frame: np.ndarray,
                    results: List[ObjectIdentifierObject],
                    overlay: bool) -> np.ndarray:
        """
        """
        frame = frame * 0 if overlay else frame
        frame_height, frame_width = frame.shape[:2]
        # Drawing parameters
        font_size = (1.5 * frame_width) / 1664
        thickness = int((2 * frame_width) / 1664)
        thickness = 1 if thickness == 0 else thickness
        padding = int((15 * frame_width) / 1664)
        line_type = cv2.LINE_AA
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        for result in results:
            class_ = result.class_name
            score = result.score
            bounds = result.bbox
            x1 = bounds.left
            y1 = bounds.top
            x2 = bounds.right
            y2 = bounds.bottom
            # x, y, w, h = bounds
            ind = self.class_names.index(class_)
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
            class_size = cv2.getTextSize(class_.lower(),
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

            top_text_background = y1 - int(class_size[0][1])
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
        return frame


__all__ = 'Drawer',

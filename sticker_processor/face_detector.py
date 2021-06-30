import numpy as np
from facenet_pytorch import MTCNN
import cv2

class FaceDetector:
    def __init__(self, min_score=0.8):
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
        self.min_score = min_score


    def get_biggest_bbox(self, bboxes):
        top_bbox = bboxes[0]
        bbox_size = 0

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            height = x2 - x1
            width = y2 - y1

            bbox_size_ = height * width
            if bbox_size_ > bbox_size:
                bbox_size = bbox_size_
                top_bbox = bbox
            
        return top_bbox


    def crop_face(self, image, bbox):
        x1, y1, x2, y2 = bbox
        height = x2 - x1
        width = y2 - y1

        crop_size = int(max(height, width) * 1.7)

        x = (x1 + x2) // 2
        y = (y1 + y2) // 2

        x1 = max(x - (crop_size // 2), 0)
        x2 = x + (crop_size // 2)

        y1 = max(y - (crop_size // 2), 0)
        y2 = y + (crop_size // 2)

        face = image[y1 : y2, x1 : x2]

        padded_face = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

        offset_y = (crop_size - face.shape[0]) // 2
        offset_x = (crop_size - face.shape[1]) // 2

        padded_face[offset_y : face.shape[0] + offset_y,
                    offset_x : face.shape[1] + offset_x] = face
        padded_face = cv2.resize(padded_face, (512, 512))

        return padded_face


    def detect(self, image):
        bboxes, scores = self.mtcnn.detect(image)
        bboxes = bboxes.astype(np.int32)
        bboxes, scores = bboxes.tolist(), scores.tolist()
        
        bboxes = [b for b, s in zip(bboxes, scores) if s >= self.min_score]
        del scores

        bbox = self.get_biggest_bbox(bboxes)
        face = self.crop_face(image, bbox)

        return face
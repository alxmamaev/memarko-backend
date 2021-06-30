import torch
import cv2
import numpy as np
import cv2 as cv
from face_segmentation import BiSeNet
import torchvision.transforms as transforms
from sticker import Sticker, Segment, SegmentElipse
import numpy as np
from sklearn.cluster import KMeans

class FaceProcessor:
    def __init__(self, model_path):
        self.segmentator = BiSeNet(n_classes=19)

        self.segmentator.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.segmentator.eval()

        self.segment2id = {
            "left_eye": 4, 
            "right_eye": 5,
            
            "left_eyebrow": 2,
            "right_eyebrow": 3, 
            "nose": 10,
            "top_lip":12,
            "bottom_lip": 13,
            "right_ear": 7, 
            "left_ear": 8, 
            "face": 1,
            "hair": 17,
        }

        self.segments_with_texture = ["hair", "face"]

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    

    def extract_polygon(self, mask):
        mask = np.expand_dims(mask.astype(np.uint8) * 255, 2) 

        _, thresh = cv2.threshold(mask, 127, 255,0)
        contours, _ = cv2.findContours(thresh, 1, 2)
        
        contour = sorted(contours, key = lambda x: x.shape[0], reverse=True)

        if not contour:
            return None

        contour = contour[0]
        epsilon = 0.005 * cv.arcLength(contour, True)
        contour = cv.approxPolyDP(contour, epsilon, True)[:,0,:]

        return contour

    
    

    def extract_texture(self, mask, image):
        edges = cv2.Canny(image, 120, 170)
        edges[~mask] = 0
        
        ret, thresh = cv2.threshold(edges, 127, 255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        contours_ = sorted(contours, key = lambda x: x.shape[0], reverse=True)[:40]
        
        contours = []
        for cnt in contours_:
            epsilon = 0.005 * cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, epsilon, True)[:,0,:]
            contours.append(cnt)

        return contours


    def segmentate(self, face_image):
        image = self.to_tensor(face_image)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            out = self.segmentator(image)[0]

        segments = out.squeeze(0).cpu().numpy().argmax(0)
        return segments


    def extract_color(self, mask, image):
        # image = cv2.blur(image, (10,10)) 
        def make_histogram(cluster):
            """
            Count the number of pixels in each cluster
            :param: KMeans cluster
            :return: numpy histogram
            """
            numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
            hist, _ = np.histogram(cluster.labels_, bins=numLabels)
            hist = hist.astype('float32')
            hist /= hist.sum()
            return hist


        def make_bar(height, width, color):
            """
            Create an image of a given color
            :param: height of the image
            :param: width of the image
            :param: BGR pixel values of the color
            :return: tuple of bar, rgb values, and hsv values
            """
            bar = np.zeros((height, width, 3), np.uint8)
            bar[:] = color
            red, green, blue = int(color[2]), int(color[1]), int(color[0])
            hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv_bar[0][0]
            return bar, (red, green, blue), (hue, sat, val)


        def sort_hsvs(hsv_list):
            """
            Sort the list of HSV values
            :param hsv_list: List of HSV tuples
            :return: List of indexes, sorted by hue, then saturation, then value
            """
            bars_with_indexes = []
            for index, hsv_val in enumerate(hsv_list):
                bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
            bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
            return [item[0] for item in bars_with_indexes]

        image = image[mask]

        if image.shape[0] == 0:
            return (1, 1, 1)
        

        # we'll pick the 5 most common colors
        num_clusters = 3
        clusters = KMeans(n_clusters=num_clusters)
        clusters.fit(image)

        # count the dominant colors and put them in "buckets"
        histogram = make_histogram(clusters)
        # then sort them, most-common first
        combined = zip(histogram, clusters.cluster_centers_)
        combined = sorted(combined, key=lambda x: x[0], reverse=True)

        # finally, we'll output a graphic showing the colors in order
        rgbs = []
        hsv_values = []
        for index, rows in enumerate(combined):
            bar, rgb, hsv = make_bar(100, 100, rows[1])
            hsv_values.append(hsv)
            rgbs.append(rgb)

        # sort the bars[] list so that we can show the colored boxes sorted
        # by their HSV values -- sort by hue, then saturation
        sorted_rgb_indexes = sort_hsvs(hsv_values)
        sorted_rgb = [rgbs[idx] for idx in sorted_rgb_indexes]

        return tuple(i / 255 for i in sorted_rgb[0])[::-1]


    @staticmethod
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def image2sticker(self, image):
        image = self.white_balance(image)
        segmentation_mask = self.segmentate(image)
        sticker = Sticker()

        pupil_size = None

        for segment_name, segment_id in self.segment2id.items():
            mask = segmentation_mask == segment_id

            color = self.extract_color(mask, image)

            polygon = self.extract_polygon(mask)
            texture = None

            if segment_name in ["left_eye", "right_eye"]:
                color = (1, 1, 1)
                
                if polygon is not None:
                    pupil_position = tuple(int(i) for i in  polygon.mean(axis=0).tolist())
                    if pupil_size is None:
                        pupil_scale = (polygon[:,1].max()-polygon[:,1].min()) / 17
                        pupil_size = int(17 * pupil_scale)

                    elipse_segment = SegmentElipse(pupil_position, (pupil_size, pupil_size))
                    sticker.segments[segment_name.replace("eye", "pupil")] = elipse_segment

            if segment_name in self.segments_with_texture:
                texture = self.extract_texture(mask, image)

            segment = Segment(polygon, texture=texture, color=color)
            sticker.segments[segment_name] = segment

            


        return sticker
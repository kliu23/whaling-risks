import cv2 as cv
import imutils
import numpy as np
from PIL import Image, ImageEnhance


class ManualScanner:
    def __init__(self, image_path):
        self.image_path = image_path
        self.manual_points = []
        self.img = cv.imread(image_path)
    
    def click_event(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.manual_points.append((x, y))

            cv.circle(self.img, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
            cv.putText(
                self.img, 
                f"Point {len(self.manual_points)}", 
                (x, y), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 0), 2
            )

            cv.imshow('image', self.img)
        

    def perspective_transform(self):
        warped = self.four_point_transform(np.array(self.manual_points))
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

        # Enhance the image
        enhancer = ImageEnhance.Contrast(Image.fromarray(warped))
        enhanced = enhancer.enhance(2.0)
        enhanced = np.array(enhanced)

        return enhanced

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


    def four_point_transform(self, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(self.img, M, (maxWidth, maxHeight))

        return warped

    

    def run(self):
        cv.imshow('image', self.img)
        cv.setMouseCallback('image', self.click_event)

        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or (len(self.manual_points) == 4 and key == ord('c')):
                break
        
        warped = self.perspective_transform()
        # cv.imshow('Warped', warped)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return warped



if __name__ == "__main__":
    image_path = "images/1866UMMIC7847_1.jpeg"

    scanner = ManualScanner(image_path)
    points = scanner.run()
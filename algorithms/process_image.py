import cv2
import numpy as np
from copy import deepcopy


class ProcessImage:
    def __init__(self, image_file):
        self.__image_file = image_file
        self.__original_image = cv2.imread(self.__image_file)
        self.__total_image_area = self.__original_image.shape[1]*self.__original_image.shape[0]
        self.__resized_image = None

    def __resize_image(self):
        ratio_ = self.__original_image.shape[1] / float(self.__original_image.shape[0])
        fixed_height = 500
        width = int(float(fixed_height) / ratio_)
        self.__total_image_area = fixed_height * width

        self.__resized_image = cv2.resize(self.__original_image, (fixed_height, width))

    def __get_contours(self):
        if self.__resized_image is not None:
            gray_image = cv2.cvtColor(self.__resized_image, cv2.COLOR_BGR2GRAY)
            smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=0)

            edge_image = cv2.Canny(smoothed_image, 100, 200)

            contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            return contours, hierarchy
        else:
            return None, None

    def __check_contours(self, contours):
        max_area = 0
        max_area_cnt = None

        for cnt in contours:

            area = cv2.contourArea(cnt)
            # print(len(cnt), area)
            if area > max_area:
                max_area = area
                max_area_cnt = cnt
                # print(len(cnt), 'max area', area )

        # print('Max contour area percentage {0}'.format(float(max_area / self.__total_image_area)))

        if max_area > self.__total_image_area * 0.5:
            return max_area_cnt
        else:
            return None

    def __perspective_correction(self, contour):
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # rectImage = cv2.drawContours(resizedImage, [box], 0, (0, 0, 255), 2)

        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # print(approx)
        # print(drawContourList)

        if len(approx) == 4:
            # contourImage = cv2.drawContours(self.__resized_image, approx, -1, (0, 255, 0), 4)

            # realPoints = approx * originalImage.shape[1]/float(fixed_height)
            real_points = approx
            real_points = np.reshape(real_points, (4, 2))

            ordered_points_ = self.order_points(real_points)
            # print(orderedPoints)

            warped_image = self.four_point_transform(self.__resized_image, ordered_points_)
            # cv2.imshow('contour image', contourImage)
            # cv2.imshow('final image', warpedImage)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imwrite('output.jpg', warpedImage)
        else:
            warped_image = self.__resized_image

        return warped_image

    @staticmethod
    def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = ProcessImage.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    @staticmethod
    def threshold_image(image):
        """
        This function take one argument as
        input. this function will convert
        input image to binary image
        :param image: image
        :return: thresholded image
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_image = image
        # converting it to binary image
        # ret, threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # ret, threshold_img = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                              11, 6)
        # saving image to view threshold image
        # kernel = np.ones((2, 2), np.uint8)
        # erosion = cv2.erode(img, kernel, iterations=1)
        # morphologicalTransfromedImage = cv2.dilate(threshold_img, kernel, iterations=1)
        # kernel = np.ones((3, 3), np.uint8)
        # morphologicalTransfromedImage = cv2.morphologyEx(morphologicalTransfromedImage, cv2.MORPH_CLOSE, kernel)
        # morphologicalTransfromedImage = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel, iterations=2)

        # cv2.imwrite('thresholded.png', threshold_img)

        # cv2.imshow('threshold image', threshold_img)
        # cv2.imshow('MT image', morphologicalTransfromedImage)
        # Maintain output window until
        # user presses a key
        # cv2.waitKey(0)
        # Destroying present windows on screen
        # cv2.destroyAllWindows()

        return 255 - threshold_img
        # return 255 - morphologicalTransfromedImage

    def get_processed_image(self):
        self.__resize_image()
        contours, hierarchy = self.__get_contours()

        if contours is not None:
            max_area_contour = self.__check_contours(contours=contours)
            if max_area_contour is not None:
                warped_image = self.__perspective_correction(contour=max_area_contour)
            else:
                warped_image = self.correct_rotation(self.__resized_image)
                # warped_image = self.__resized_image

            thresholded_image = self.threshold_image(warped_image)
        else:
            rotation_corrected_image = self.correct_rotation(self.__resized_image)
            thresholded_image = self.threshold_image(rotation_corrected_image)

        return thresholded_image

    def slope(self, x1, y1, x2, y2):
        if x1 == x2:
            return 0
        slope = (y2 - y1) / (x2 - x1)
        theta = np.rad2deg(np.arctan(slope))
        return theta

    def rotate(self, img, theta):
        rows, cols = img.shape[0], img.shape[1]
        image_center = (cols / 2, rows / 2)

        M = cv2.getRotationMatrix2D(image_center, theta, 1)

        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        bound_w = int(rows * abs_sin + cols * abs_cos)
        bound_h = int(rows * abs_cos + cols * abs_sin)

        M[0, 2] += bound_w / 2 - image_center[0]
        M[1, 2] += bound_h / 2 - image_center[1]

        # rotate orignal image to show transformation
        rotated = cv2.warpAffine(img, M, (bound_w, bound_h), borderValue=(255, 255, 255))
        return rotated

    def correct_rotation(self, image):
        print('rotation correction applied')
        # print(type(image))
        img = deepcopy(image)
        # print(type(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 6)
        # edges = 255 - edges
        kernel = np.ones((3, 3), np.uint8)
        # edges = cv2.erode(edges, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # edges = cv2.erode(edges, kernel, iterations=1)

        # cv2.imshow('edges', edges)

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        angles = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            theta = self.slope(box[0][0], box[0][1], box[1][0], box[1][1])
            if abs(theta) < 45:
                angles.append(theta)
                # print(theta)

        orientation = sum(angles)/len(angles)
        print('Orientation Correction', orientation)
        finalImage = self.rotate(image, orientation)

        # cv2.imshow('contours', img)

        # minLineLength = 100
        # maxLineGap = 100
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # # print(lines[0])
        # for i in lines:
        #     # print(i)
        #     x1, y1, x2, y2 = i[0]
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        # # print(len(lines))
        # for line in lines:
        #     rho, theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imshow('lines', img)
        # cv2.imshow('rotated', finalImage)
        # cv2.waitKey()
        return finalImage


if __name__ == '__main__':
    process_image = ProcessImage('../test_data/1602305321.jpg')
    thresholded_image = process_image.get_processed_image()

    cv2.imshow('thresholded image', thresholded_image)
    cv2.waitKey(0)
    # cv2.destroyWindow()

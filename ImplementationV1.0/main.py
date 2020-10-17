import cv2
import csv
import os
import numpy as np

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


def pre_processing(image):
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
    threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
    # saving image to view threshold image
    # kernel = np.ones((2, 2), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # morphologicalTransfromedImage = cv2.dilate(threshold_img, kernel, iterations=1)
    # kernel = np.ones((3, 3), np.uint8)
    # morphologicalTransfromedImage = cv2.morphologyEx(morphologicalTransfromedImage, cv2.MORPH_CLOSE, kernel)
    # morphologicalTransfromedImage = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # cv2.imwrite('thresholded.png', threshold_img)

    cv2.imshow('threshold image', threshold_img)
    # cv2.imshow('MT image', morphologicalTransfromedImage)
    # Maintain output window until
    # user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()

    return 255 - threshold_img
    # return 255 - morphologicalTransfromedImage


def parse_text(threshold_img):
    """
    This function take one argument as
    input. this function will feed input
    image to tesseract to predict text.
    :param threshold_img: image
    return: meta-data dictionary
    """
    # configuring parameters for tesseract
    tesseract_config = r'--oem 3 --psm 6'
    # now feeding image to tesseract
    details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT,
                                        config=tesseract_config, lang='eng')
    return details


def draw_boxes(image, details, threshold_point):
    """
    This function takes three argument as
    input. it draw boxes on text area detected
    by Tesseract. it also writes resulted image to
    your local disk so that you can view it.
    :param image: image
    :param details: dictionary
    :param threshold_point: integer
    :return: None
    """
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > threshold_point:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # saving image to local
    cv2.imwrite('captured_text_area.png', image)
    # display image
    cv2.imshow('captured text', image)
    # Maintain output window until user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()


def format_text(details):
    """
    This function take one argument as
    input.This function will arrange
    resulted text into proper format.
    :param details: dictionary
    :return: list
    """
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word != '':
            word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)
            word_list = []

    return parse_text


def write_text(formatted_text):
    """
    This function take one argument.
    it will write arranged text into
    a file.
    :param formatted_text: list
    :return: None
    """

    with open('resulted_text.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)

    output = None
    with open('resulted_text.txt', 'r') as f:
        output = f.read()

    return output


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


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
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


def do_the_thing(filename):
    originalImage = cv2.imread(filename)
    ratio_ = originalImage.shape[1]/float(originalImage.shape[0])
    fixed_height = 500
    width = int(float(fixed_height)/ratio_)
    totalImageArea = fixed_height*width
    resizedImage = cv2.resize(originalImage, (fixed_height, width))
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    smoothedImage = cv2.GaussianBlur(grayImage, (5, 5), sigmaX=0)

    edgeImage = cv2.Canny(smoothedImage, 100, 200)

    contours, hierarchy = cv2.findContours(edgeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('Edge Image', edgeImage)
    # cv2.imshow('Original image', originalImage)
    # cv2.imshow('Gray image', grayImage)

    max_area = 0
    max_area_cnt = None

    for cnt in contours:

        area = cv2.contourArea(cnt)
        # print(len(cnt), area)
        if area > max_area:
            max_area = area
            max_area_cnt = cnt
            # print(len(cnt), 'max area', area )

    print('Max contour area percentage {0}'.format(float(max_area/totalImageArea)))
    if max_area > totalImageArea * 0.5:
        cnt = max_area_cnt
        len(max_area_cnt)
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # rectImage = cv2.drawContours(resizedImage, [box], 0, (0, 0, 255), 2)

        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # print(approx)
        # print(drawContourList)

        if len(approx) == 4:
            contourImage = cv2.drawContours(resizedImage, approx, -1, (0, 255, 0), 4)

            # realPoints = approx * originalImage.shape[1]/float(fixed_height)
            realPoints = approx
            realPoints = np.reshape(realPoints, (4, 2))

            orderedPoints = order_points(realPoints)
            # print(orderedPoints)

            warpedImage = four_point_transform(resizedImage, orderedPoints)
            # cv2.imshow('contour image', contourImage)
            # cv2.imshow('final image', warpedImage)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imwrite('output.jpg', warpedImage)
        else:
            warpedImage = resizedImage
    else:
        warpedImage = resizedImage

    thresholds_image = pre_processing(warpedImage)
    # calling parse_text function to get text from image by Tesseract.
    parsed_data = parse_text(thresholds_image)
    print(parsed_data)
    # defining threshold for draw box
    accuracy_threshold = 30
    # calling draw_boxes function which will draw dox around text area.
    draw_boxes(thresholds_image, parsed_data, accuracy_threshold)
    # calling format_text function which will format text according to input image
    arranged_text = format_text(parsed_data)
    # calling write_text function which will write arranged text into file
    return write_text(arranged_text)


if __name__ == '__main__':
    special_check = '1599028073'
    with open('total_results.txt', 'w') as f:
        for root, dirs, files in os.walk("../../Data", topdown=False):
            for name in files:
                if special_check in name:
                    path = os.path.join(root, name)
                    s = 'File path {0}\n'.format(path)
                    s += do_the_thing(path)
                    print(s)
                    f.write(s + '\n')
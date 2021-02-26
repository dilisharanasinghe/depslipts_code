import pytesseract
import cv2
import csv

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


class ExtractCharacters:
    def __init__(self, thresholded_image):
        self.__image = thresholded_image

    def extract(self):
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
        details = pytesseract.image_to_data(self.__image, output_type=pytesseract.Output.DICT,
                                            config=tesseract_config, lang='eng')
        return details

    @staticmethod
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
        cv2.imwrite('../captured_text_area.png', image)
        # display image
        # cv2.imshow('captured text', image)
        # Maintain output window until user presses a key
        # cv2.waitKey(0)
        # Destroying present windows on screen
        # cv2.destroyAllWindows()

    @staticmethod
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

    @staticmethod
    def write_text(formatted_text):
        """
        This function take one argument.
        it will write arranged text into
        a file.
        :param formatted_text: list
        :return: None
        """

        with open('../resulted_text.txt', 'w', newline="") as file:
            csv.writer(file, delimiter=" ").writerows(formatted_text)

        output = None
        with open('../resulted_text.txt', 'r') as f:
            output = f.read()

        return output
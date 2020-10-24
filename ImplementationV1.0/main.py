import os
import cv2
from check_slip import CheckSlip
from process_image import ProcessImage
from extract_characters import ExtractCharacters


def do_the_thing(filename):
    process_image = ProcessImage(image_file=filename)
    thresholded_image = process_image.get_processed_image()

    extract_characters = ExtractCharacters(thresholded_image=thresholded_image)
    parsed_text = extract_characters.extract()

    # print(parsed_text)
    check_slip = CheckSlip(parsed_text=parsed_text)
    check_slip.check_account_number('016210003383')
    check_slip.check_payment(amount=2500)
    check_slip.check_transaction_number()

    cv2.imshow('thresholded image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
    special_check = None  # '1599152317.jpg'
    with open('total_results.txt', 'w') as f:
        for root, dirs, files in os.walk("../../Data", topdown=False):
            for name in files:
                if special_check is not None:
                    if special_check in name:
                        path = os.path.join(root, name)
                        s = 'File path {0}\n'.format(path)
                        print(s)
                        do_the_thing(path)
                else:
                    path = os.path.join(root, name)
                    s = 'File path {0}\n'.format(path)
                    print(s)
                    do_the_thing(path)

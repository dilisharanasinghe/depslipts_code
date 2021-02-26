import numpy as np
import re


class CheckSlip:
    def __init__(self, parsed_text):
        self.__parsed_text = parsed_text
        self.__transaction_identifiers = ['refno', 'trans no', 'trans no.',
                                          'trans .id', 'trx id', 'reference no', 'trans']

        self.__payment_identifiers = ['total', 'deposits']

    @staticmethod
    def levenshtein_distance(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        # print (matrix)
        return matrix[size_x - 1, size_y - 1]

    def get_row_values(self, match_location, threshold=10):
        parsed_data_location = match_location[0]
        confidence = match_location[2]
        left = match_location[3]
        top = match_location[4]

        total_count = len(self.__parsed_text['text'])
        possible_texts = []
        for i in range(parsed_data_location, total_count):
            # print(self.__parsed_text['left'][i], self.__parsed_text['top'][i])
            if abs(top - self.__parsed_text['top'][i]) < threshold:
                # print(self.__parsed_text['text'][i])
                possible_texts.append([self.__parsed_text['text'][i],
                                      self.__parsed_text['conf'][i]])
                # print(self.__parsed_text['conf'][i])

        return possible_texts

    def check_account_number(self, account_number):
        account_number_verified = False

        for text in self.__parsed_text['text']:
            if not account_number_verified:
                if self.levenshtein_distance(str(account_number), text) == 0.0:
                    account_number_verified = True
                    break

        # print('account number verified', account_number_verified)
        return account_number_verified

    def check_payment(self, amount):
        min_distance, best_match = self.extract_matches(self.__payment_identifiers)
        # print('transaction check', min_distance, best_match)

        payment_verified = False
        if best_match is not None:
            possible_values = self.get_row_values(best_match)

            ss = ''
            for i in possible_values:
                ss += i[0]

            rr = re.findall("[-+]?[0]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", ss)
            # print(rr)

            if len(rr) > 0:
                if amount == float(rr[0].replace(',','')):
                    # print('payment verified')
                    payment_verified = True
        return payment_verified

    def check_transaction_number(self):
        min_distance, best_match = self.extract_matches(self.__transaction_identifiers)
        # print('transaction check', min_distance, best_match)

        possible_trans_no_parts = []
        confidence = 100
        if best_match is not None:
            possible_values = self.get_row_values(best_match)
            # print(possible_values)

            for value in possible_values:
                have_number = bool(re.search(r'\d', value[0]))

                if have_number:
                    possible_trans_no_parts.append(value[0])
                    if int(value[1]) < confidence and int(value[1]) != -1:
                        confidence = int(value[1])

        # print(possible_trans_no_parts, confidence)
        ss = ''
        for i in possible_trans_no_parts:
            ss += i

        return ss, confidence

    def extract_matches(self, identifiers):
        text_count = len(self.__parsed_text['text'])
        min_distance = np.inf
        best_match = None

        for i in range(text_count):
            text = self.__parsed_text['text'][i]

            for identifier in identifiers:
                l_distance = self.levenshtein_distance(text.lower(), identifier)

                if l_distance < min_distance and l_distance <= 1.0:
                    min_distance = l_distance
                    details = self.__parsed_text
                    best_match = (i, details['text'][i], details['conf'][i],
                                  details['left'][i], details['top'][i],
                                  details['width'][i], details['height'][i])

        return min_distance, best_match


if __name__ == '__main__':
    parsed_data = {'level': [1, 2, 3, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5], 'page_num': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'block_num': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'par_num': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'line_num': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22], 'word_num': [0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2], 'left': [0, 20, 20, 204, 204, 266, 404, 204, 204, 270, 204, 204, 285, 38, 38, 65, 110, 200, 254, 137, 137, 182, 25, 25, 166, 181, 285, 25, 25, 165, 181, 23, 23, 165, 180, 302, 23, 23, 60, 164, 180, 123, 123, 174, 122, 122, 56, 56, 137, 167, 259, 93, 93, 159, 177, 20, 20, 65, 121, 158, 204, 251, 309, 347, 20, 20, 67, 121, 157, 203, 260, 308, 349, 21, 21, 76, 121, 157, 22, 22, 75, 159, 174, 211, 23, 23, 49, 75, 158, 192, 24, 24, 68, 125, 129, 26, 26, 77, 123, 138, 95, 95, 156, 23, 23, 119], 'top': [0, 0, 0, 0, 18, 18, 0, 18, 33, 18, 50, 43, 50, 74, 87, 85, 80, 78, 74, 92, 97, 92, 121, 134, 133, 125, 121, 145, 153, 151, 145, 156, 173, 169, 160, 156, 183, 192, 189, 187, 183, 219, 224, 219, 262, 262, 278, 291, 289, 284, 278, 323, 331, 330, 323, 359, 376, 372, 374, 369, 366, 362, 365, 359, 379, 395, 391, 394, 388, 385, 381, 384, 379, 407, 413, 410, 413, 407, 439, 448, 445, 447, 443, 439, 461, 466, 465, 465, 465, 461, 479, 500, 488, 488, 479, 508, 516, 516, 517, 508, 546, 546, 546, 543, 582, 546], 'width': [408, 388, 388, 204, 54, 12, 4, 119, 56, 53, 120, 71, 39, 324, 17, 30, 80, 40, 108, 135, 32, 90, 331, 82, 2, 92, 71, 226, 71, 3, 70, 335, 54, 2, 112, 56, 190, 26, 44, 3, 33, 120, 39, 69, 120, 120, 273, 70, 19, 80, 70, 206, 55, 7, 122, 336, 21, 35, 7, 7, 22, 37, 7, 9, 332, 21, 33, 7, 8, 23, 27, 8, 3, 144, 20, 24, 6, 8, 263, 40, 62, 3, 22, 74, 177, 16, 16, 43, 3, 8, 353, 33, 47, 2, 248, 338, 41, 16, 2, 226, 68, 7, 7, 263, 71, 60], 'height': [630, 584, 584, 30, 12, 11, 4, 32, 17, 28, 17, 31, 13, 27, 14, 14, 17, 14, 15, 20, 15, 17, 33, 20, 10, 19, 18, 27, 19, 10, 17, 34, 17, 10, 20, 18, 26, 17, 18, 11, 16, 25, 20, 22, 24, 24, 35, 22, 19, 22, 23, 27, 19, 16, 22, 32, 15, 16, 12, 14, 16, 17, 11, 15, 31, 15, 17, 11, 15, 16, 17, 12, 16, 21, 15, 16, 11, 15, 24, 15, 18, 9, 15, 19, 19, 14, 14, 14, 9, 14, 34, 13, 25, 22, 32, 21, 13, 12, 9, 19, 13, 13, 12, 41, 2, 38], 'conf': ['-1', '-1', '-1', '-1', 0, 9, 0, '-1', 10, 33, '-1', 95, 88, '-1', 72, 90, 74, 92, 45, '-1', 96, 74, '-1', 82, 57, 81, 62, '-1', 59, 61, 64, '-1', 71, 0, 44, 95, '-1', 36, 88, 50, 88, '-1', 88, 91, '-1', 82, '-1', 25, 96, 95, 95, '-1', 33, 92, 82, '-1', 91, 88, 92, 72, 84, 95, 88, 88, '-1', 88, 93, 89, 89, 73, 34, 88, 87, '-1', 78, 93, 91, 73, '-1', 95, 87, 22, 87, 74, '-1', 96, 95, 96, 85, 89, '-1', 19, 13, 63, 61, '-1', 81, 61, 59, 39, '-1', 61, 69, '-1', 17, 11], 'text': ['', '', '', '', 'ade', 'OF', ':', '', 'saab', 'Saba', '', 'Sampath', 'Bonk', '', 'NO', '268,', 'KOLONNAHA', 'ROAD,', 'WELLAMPITIYA', '', 'TEL:', '0112533498', '', 'DATE/TIME', '>', '2020/10/08', '14:47:19', '', 'TERMINAL', ':', 'K0004521', '', 'BRANCH', ':', 'WELLAMPITIYA', 'BRANCH', '', 'pH', 'TRACE', '2', '2861', '', 'CASH', 'DEPOSIT', '', '016210003383', '', 'AGADEMY', 'OF', 'ABSOLUTE', 'SCIENCE', '', 'AWOUNT', '&', 'DENOMINATIONS', '', 'Rs.', '5000', 'x', '0', 'Bs.', '2000', 'x', '0', '', 'Rs.', '1000', 'x', '2', 'Rs.', '800', 'x', 'J', '', 'Rs,', '100', 'x', '0', '', 'Total', 'Deposit', ':', 'Rs,', '2,500.00', '', 'No', 'of', 'Notes', ':', '3', '', 'WOT', 'IRME', '7', '201282', '', 'TRANS', 'NO.', ':', 'KO00452120201008114719-02', '', '1', 't', '', 'ce', 'eno']}

    check_slip = CheckSlip(parsed_text=parsed_data)
    check_slip.check_account_number('016210003383')
    check_slip.check_payment(amount=2500)
    check_slip.check_transaction_number()
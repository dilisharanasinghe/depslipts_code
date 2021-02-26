import mysql.connector


class DatabaseConnector:
    def __init__(self):
        self.__my_db = mysql.connector.connect(host="localhost",
                                               user="dilshan",
                                               password="123456",
                                               database='depslips')

        print(self.__my_db)

    def insert_data(self, record):
        my_cursor = self.__my_db.cursor()

        sql = "INSERT INTO deposit_slip_records (user_id, slip_id, image_url, account_number, amount," \
              "account_number_verified, amount_verified, transaction_id, confidence_value, confidence_level" \
              ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (record['userId'],
               record['slipId'],
               record['imageUrl'],
               record['accountNumber'],
               record['amount'],
               record['accountNumberVerified'],
               record['amountVerified'],
               record['transactionId'],
               record['confidenceValue'],
               record['confidenceLevel'])

        my_cursor.execute(sql, val)

        self.__my_db.commit()
        my_cursor.close()

    def get_data(self):
        my_cursor = self.__my_db.cursor()
        my_cursor.execute("SELECT * FROM deposit_slip_records")
        my_result = my_cursor.fetchall()

        for x in my_result:
            print(x)

        my_cursor.close()


if __name__ == '__main__':
    db_c = DatabaseConnector()
    db_c.get_data()
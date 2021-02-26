from abc import ABCMeta
import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web
from multiprocessing import Process, Queue
from algorithms.process_slip import ProcessSlip
from database.database_connector import DatabaseConnector
import json
import os


queue_request = Queue()


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/upload", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)


class UploadHandler(tornado.web.RequestHandler, metaclass=ABCMeta):
    def post(self):
        try:
            slip_id = self.request.headers['SlipId']
            user_id = self.request.headers['UserId']
            image_url = self.request.headers['ImageUrl']
            amount = self.request.headers['Amount']
            account_number = self.request.headers['Account-Number']

            queue_request.put({'slipId': slip_id,
                               'userId': user_id,
                               'imageUrl': image_url,
                               'amount': amount,
                               'accountNumber': account_number})

            self.write(json.dumps({'success': True, 'message': 'Request Queued'}))
        except Exception as e:
            self.write(json.dumps({'success': False, 'message': str(e)}))
            self.set_status(400)


class ProcessQueue(Process):
    def __init__(self, process_number=0, queue_input=Queue()):
        super(ProcessQueue, self).__init__()
        self.__process_number = process_number
        self.__process_slip = ProcessSlip()
        self.__queue_input = queue_input

    def run(self):
        db_connector = DatabaseConnector()
        while True:
            try:
                request = self.__queue_input.get()
                file_name = 'temp_images/temp_image_{0}.jpg'.format(self.__process_number)
                # urllib.request.urlretrieve(request['imageUrl'], file_name)
                result = self.__process_slip.do_the_thing(filename=file_name,
                                                          amount=int(request['amount']),
                                                          account_number=request['accountNumber'])
                print(result)

                request['accountNumberVerified'] = result['account_verified']
                request['amountVerified'] = result['amount_verified']
                request['transactionId'] = result['transaction_number']
                request['confidenceValue'] = result['confidence']
                request['confidenceLevel'] = 'MEDIUM' if 33 < result['confidence'] < 66 else 'LOW' if result['confidence'] <= 33 else 'HIGH'

                db_connector.insert_data(record=request)
                # os.remove(file_name)
            except Exception as e:
                pass


def run_server():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8080)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    process_queue = ProcessQueue(process_number=0, queue_input=queue_request)
    process_queue.start()

    run_server()
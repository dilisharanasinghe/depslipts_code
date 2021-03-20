from abc import ABCMeta
import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web
from multiprocessing import Process, Queue
from algorithms.process_slip import ProcessSlip
# from database.database_connector import DatabaseConnector
from requests.feedback import ProcessingDone
import json
import os
import urllib.request


queue_request = Queue()


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/slips", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)


class UploadHandler(tornado.web.RequestHandler, metaclass=ABCMeta):
    def post(self):
        try:
            json_msg = json.loads(self.request.body)
            slip_id = json_msg['slipId']
            image_url = json_msg['imageUrl']
            transaction_value = json_msg['transactionValue']
            account_number = json_msg['accountNumber']
            token = json_msg['token']

            queue_request.put({'slipId': slip_id,
                               'imageUrl': image_url,
                               'transactionValue': transaction_value,
                               'accountNumber': account_number,
                               'token': token})

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
        # db_connector = DatabaseConnector()
        while True:
            try:
                request = self.__queue_input.get()
                print('-' * 50)
                print(request)
                file_name = 'temp_images/temp_image_{0}.jpg'.format(self.__process_number)
                try:
                    # print(request['imageUrl'])
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)

                    urllib.request.urlretrieve(request['imageUrl'], file_name)
                except Exception as e:
                    print(e)
                    continue
                print('downloaded')

                result = self.__process_slip.do_the_thing(filename=file_name,
                                                          amount=int(request['transactionValue']),
                                                          account_number=request['accountNumber'])

                print(result)
                print('-' * 50)

                request_body = {'slipId': request['slipId'],
                                'confidenceLevel': 'MEDIUM' if 33 < result['confidence'] < 66 else 'LOW' if result['confidence'] <= 33 else 'HIGH',
                                'confidenceValue': result['confidence'],
                                'accountNumberVerified': result['account_verified'],
                                'transactionValueVerified': result['amount_verified'],
                                'transactionId': result['transaction_number']}

                ProcessingDone.patch({'token': request['token'],
                                      'slipId': request['slipId'],
                                      'body': json.dumps(request_body)})

                # db_connector.insert_data(record=request)
                os.remove(file_name)
            except Exception as e:
                print('Exception ', e)
                # raise e
                pass


def run_server():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8080)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    process_queue = ProcessQueue(process_number=0, queue_input=queue_request)
    process_queue.start()

    run_server()
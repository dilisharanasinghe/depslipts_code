from tornado.httpclient import HTTPClient, HTTPRequest, HTTPError


class ProcessingDone:
    def __init__(self):
        pass

    @staticmethod
    def patch(data):
        try:
            request = HTTPRequest(url='http://www.deposits.egosurf.lk/ml/slips/{0}'.format(data['slipId']),
                                  method='PATCH',
                                  headers={'Authorization': 'Bearer {0}'.format(data['token'])},
                                  body=data['body'])
            http_client = HTTPClient()

            response = http_client.fetch(request)
        except HTTPError as e:
            print('Processing done feedback error', e)
        except Exception as e:
            print('Processing done feedback error', e)


if __name__ == '__main__':
    t = ProcessingDone()
    t.patch({'token': 0,
             'slipId': '78',
             'body': '33'})

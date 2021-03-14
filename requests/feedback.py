from tornado.httpclient import HTTPClient, HTTPRequest, HTTPError


class ProcessingDone:
    def __init__(self):
        pass

    @staticmethod
    def patch(data):
        try:
            # print('data', data)
            request = HTTPRequest(url='http://deposits.egosurf.lk/ml/slips/{0}'.format(data['slipId']),
                                  method='PATCH',
                                  headers={'Authorization': 'Bearer {0}'.format(data['token'])},
                                  body=data['body'])
            http_client = HTTPClient()

            response = http_client.fetch(request)
        except HTTPError as e:
            print('Processing done feedback HTTPError', e)
        except Exception as e:
            print('Processing done feedback error', e)


if __name__ == '__main__':
    t = ProcessingDone()
    t.patch({'token': 0,
             'slipId': '78',
             'body': '33'})

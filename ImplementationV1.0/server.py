import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
from tornado.options import define, options
from process_slip import ProcessSlip
import json
import os

define("port", default=80, help="run on the given port", type=int)


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload_form.html")


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        # print(self.request.files)
        try:
            account_number = self.get_body_argument("account_number")
            amount = int(self.get_body_argument("amount"))
            # print(account_number, amount, type(account_number))
            file1 = self.request.files['file1'][0]
            original_fname = file1['filename']
            extension = os.path.splitext(original_fname)[1]
            fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
            final_filename= fname+extension
            output_file = open("uploads/" + final_filename, 'wb')
            output_file.write(file1['body'])
            output_file.close()

            result = ProcessSlip.do_the_thing('uploads/' + final_filename, account_number=account_number,
                                              amount=amount)

            os.remove('uploads/' + final_filename)
            self.finish(json.dumps(result))
        except Exception as e:
            self.finish("Bad request : " + str(e))


def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
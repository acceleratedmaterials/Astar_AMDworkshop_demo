from bottle import route, run
from predict import *
import os

@route('/<input_line>')
def index(input_line):
    return {'result': predict(input_line, 10)}

if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True)

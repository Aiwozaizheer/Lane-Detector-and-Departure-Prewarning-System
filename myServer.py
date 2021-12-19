# coding=utf-8


from typing import Dict
from flask import Flask, jsonify, request, render_template
from flask.helpers import url_for
from werkzeug.utils import redirect
from lane_detect import *
import os
# import json
from video import *

app = Flask(__name__)


# api = API(app)-

@app.route('/')
def index():
    return render_template('index-dp.html')


@app.route('/try', methods=['POST'])
def run():
    url = request.form['url']
    # while url:
    #     print(url)
    out = "out" + url
    print(os.getcwd())

    annotate_video(url, out)
    # print(os.getcwd())
    videoOut(os.getcwd() + "\\" + out)
    # return None
    return render_template('index-dp.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

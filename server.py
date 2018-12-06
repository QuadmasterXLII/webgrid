from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from binascii import a2b_base64

import time
import pickle
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def hello():
    return "<meta http-equiv=refresh content=\"0; URL='/static/index.html'\" />"


@app.route('/imageupload/<float:name>', methods=['POST'])
def imageupload(name):
    filename = secure_filename(str(name) + ".png")
    out = open(os.path.join("image_uploads", filename), "wb")
    image_enc = request.form["image"]
    header = "data:image/png;base64,"
    assert(image_enc[:len(header)] == header)
    print(len(image_enc))
    if len(image_enc) < 9000000:
        image_enc = image_enc[len(header):]
        out.write(a2b_base64(image_enc))
        out.close()
    return ''

@app.route('/uploadtrack', methods=["POST"])
def uploadtrack():
    json = request.get_json()
    pickle.dump(json, open("trajectory/" + "S"+ ".pickle", "wb"))
    pickle.dump(json, open("trajectory/" + str(time.time()) + ".pickle", "wb"))


    return ""



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, ssl_context='adhoc')

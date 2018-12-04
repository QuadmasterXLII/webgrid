from flask import Flask, request
import os
from werkzeug.utils import secure_filename
from binascii import a2b_base64
app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, ssl_context='adhoc')

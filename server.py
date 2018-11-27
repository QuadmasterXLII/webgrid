from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from binascii import a2b_base64
import registerFromImage
import numpy as np
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

@app.route('/linestotransform', methods=["POST"])
def linestotransform():
    json = request.get_json()
    #print(json)
    imu = json["imu"]
    if imu["alpha"]:
        roll, pitch, yaw = [imu[n] * 3.1415 / 180 for n in ("gamma", "beta", "alpha")]
    vertical_scale = json['shape'][1] / json['shape'][0]
    error, vector = registerFromImage.registerFromLines(np.array([[l[:2]] for l in json['lines']]), 
        attitude = imu["alpha"] and np.array([[roll], [yaw], [pitch]]), 
        vertical_factor=vertical_scale, 
        graph=False, 
        focalLength = (4/3) * 117)


    return jsonify({'error': error, "vector": [v for v in vector]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443, ssl_context=('cert.pem', 'key.pem'))
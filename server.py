from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<meta http-equiv=refresh content=\"0; URL='/static/index.html'\" />"


@app.route('/imageupload/<float:name>', methods=['POST'])
def imageupload(name):
    filename = secure_filename(str(name) + ".png")
    out = open(os.path.join("image_uploads", filename), "wb")
    out.write(request.data)
    out.close()
    return ''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443, ssl_context='adhoc')
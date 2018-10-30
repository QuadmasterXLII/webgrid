from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<meta http-equiv=refresh content=\"0; URL='/static/index.html'\" />"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443, ssl_context='adhoc')
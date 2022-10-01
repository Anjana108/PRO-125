from flask import Flask, jsonify, request
from classifier2 import prediction

app = Flask(__name__)

@app.route("/predictData", methods=["POST"])

def getPrediction():
    image = request.files.get("alphabet")
    predict = prediction(image)
    return jsonify({
        "prediction": predict
    }), 200

if (__name__ == "__main__"):
    app.run(debug = True)
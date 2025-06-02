from flask import Flask, request, jsonify
import train

app = Flask(__name__)

@app.post("/train")
def run_training():
    params = request.json or {}
    info = train.train_linear(**params)
    return jsonify(info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

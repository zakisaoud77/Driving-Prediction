from db.models import app
from flask import jsonify
from driving_prediction.prediction_app import prediction_app

app.register_blueprint(prediction_app,  url_prefix='/driving_prediction')


@app.route("/")
def helloworld():
    return "Hello World!"


if __name__ == "__main__":
    app.run()

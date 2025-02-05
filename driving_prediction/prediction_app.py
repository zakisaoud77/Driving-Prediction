from db.models import *
from driving_prediction.prediction import Prediction
from flask import Blueprint, render_template, session, jsonify
from sqlalchemy import and_
import time
prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/get_driving_type_from_sequence/<int:sequence_id>', methods=['GET'])
def get_driving_type_prediction_from_sequence(sequence_id):
    print(f"Starting driving Predicting from sequence N° {sequence_id}")
    start_time = time.time()
    prediction = Prediction()
    features = prediction.driving_sequence_to_np_array(sequence_id)
    pred_result = prediction.get_svm_prediction(features)
    print("PREDICTION : " + str(pred_result))
    print("computation time")
    print("--- %s seconds ---" % (time.time() - start_time))
    return jsonify(pred_result)


@prediction_app.route('/get_driving_type_from_user/<string:firstname>/<string:lastname>/<string:driving_time>', methods=['GET', 'POST'])
def get_driving_type_prediction_from_user(firstname, lastname, driving_time):
    print(f"Starting driving Predicting for user {firstname}-{lastname} and timestamp {driving_time}")
    start_time = time.time()
    prediction = Prediction()
    try:
        # verify if user exist
        user = User.get_user(firstname, lastname)
        if user:
            # verify if sequence exist for the timestamp
            user_id = user.id
            sequences = DrivingSequence.get_sequences_from_timestamp(driving_time)
            for sequence in sequences:
                user_driving = UserDriving.get_user_driving(user_id, sequence.id)
                if user_driving:
                    features = prediction.driving_sequence_to_np_array(sequence.id)
                    pred_result = prediction.get_svm_prediction(features)
                    print("PREDICTION : " + str(pred_result))
                    print("computation time")
                    print("--- %s seconds ---" % (time.time() - start_time))
                    pred_result_obj = DrivingPrediction(driving_sequence=sequence.id, prediction=pred_result)
                    db.session.add(pred_result_obj)
                    db.session.commit()
                    return jsonify(pred_result)
            if not user_driving:
                print("There is no driving sequence for this user in this periode of time")
                return jsonify(None)
        else:
            print(f"User {firstname}-{lastname }doesn't exist")
    except Exception as e:
        print(f"Can't get the driving sequence because of {e}")
    return jsonify(None)

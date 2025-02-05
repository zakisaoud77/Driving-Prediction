import time
import pickle
from db.models import DrivingSequence
from svm_class import *


class Prediction:
    def __init__(self):
        try:
            with open('driving_prediction/model.pkl', 'rb') as inp:
                self.svm = pickle.load(inp)
                print("Modèle chargé à partir du fichier.")
        except FileNotFoundError:
            print("ERROR : FILE_NOT_FOUND")

    def driving_sequence_to_np_array(self, sequence_id):
        try:
            driving_sequence = DrivingSequence.query.filter_by(id=sequence_id).one()
            sequence_array = np.array([float(x) for x in driving_sequence.sequence.split(",")])
            return sequence_array
        except Exception as e:
            print(f"Couldn't find the sequence due to {e}")

    def get_svm_prediction(self, x):
        result = "ERROR"
        try:
            prediction = self.svm.predict(x)
            if prediction[0] == 0:
                return "NORMAL"
            else:
                return "AGGRESSIVE"
        except Exception as e:
            print(f"Couldn't make the prediction because of {e}")

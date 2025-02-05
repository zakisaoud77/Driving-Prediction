from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Flask
from sqlalchemy import and_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://mobh_owner:bo0PXw3skqlm@ep-shy-grass-a233d1zx.eu-central-1.aws.neon.tech:5432/mobh'
db = SQLAlchemy(app)
migrate = Migrate(app, db)


class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(), nullable=False)
    lastname = db.Column(db.String(), nullable=False)
    address = db.Column(db.String())
    email = db.Column(db.String())
    drivings = db.relationship('UserDriving', back_populates='user')

    def __init__(self, firstname, lastname, email):
        self.firstname = firstname
        self.lastname = lastname
        self.email = email

    def __repr__(self):
        return f"<User {self.firstname} {self.lastname}>"

    @staticmethod
    def get_user(firstname, lastname):
        user = User.query.filter(and_(User.firstname == firstname, User.lastname == lastname)).one()
        return user


class DrivingSequence(db.Model):
    __tablename__ = 'driving_sequence'

    id = db.Column(db.Integer, primary_key=True)
    sequence = db.Column(db.String(), nullable=False)
    driving_time_begin = db.Column(db.String(), nullable=False)
    prediction = db.relationship('DrivingPrediction', back_populates='driving_sequence', uselist=False)
    users = db.relationship('UserDriving', back_populates='driving')

    def __init__(self, sequence, driving_time_begin):
        self.sequence = sequence
        self.driving_time_begin = driving_time_begin

    def __repr__(self):
        return f"<Driving Sequence {self.driving_time_begin}:{self.sequence}>"

    @staticmethod
    def get_sequences_from_timestamp(driving_time):
        sequences = DrivingSequence.query.filter_by(driving_time_begin=driving_time).all()
        return sequences


class UserDriving(db.Model):
    __tablename__ = 'user_driving'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'), nullable=False)
    driving_id = db.Column(db.Integer(), db.ForeignKey('driving_sequence.id'), nullable=False)
    driving = db.relationship('DrivingSequence', back_populates="users")
    user = db.relationship('User', back_populates="drivings")

    def __init__(self, user_id, driving_id):
        self.user_id = user_id
        self.driving_id = driving_id

    def __repr__(self):
        return f"<User Driving {self.user_id}: {self.driving_id}>"

    @staticmethod
    def get_user_driving(user_id, driving_id):
        user_driving = UserDriving.query.filter(and_(
            UserDriving.user_id == user_id, UserDriving.driving_id == driving_id)
        ).one()
        return user_driving


class DrivingPrediction(db.Model):
    __tablename__ = 'driving_prediction'

    id = db.Column(db.Integer, primary_key=True)
    driving_sequence_id = db.Column(db.Integer(), db.ForeignKey('driving_sequence.id'), nullable=False)
    driving_sequence = db.relationship("DrivingSequence", back_populates="prediction", uselist=False)
    prediction = db.Column(db.String(), nullable=False)

    def __init__(self, driving_sequence, prediction):
        self.driving_sequence_id = driving_sequence
        self.prediction = prediction

    def __repr__(self):
        return f"<Driving Prediction {self.driving_sequence}: {self.prediction}>"

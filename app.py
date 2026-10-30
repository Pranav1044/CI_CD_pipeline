import joblib


model = joblib.load("model.pkl")

# sample = [[1290,3,5,3,1980,"Downtown","Fair","Yes"]] # This exactly will not work as it require feature scaling before input
# sample_numeric = [[1290,3,5,3,1980,0,2,1]] # Converted the sample to numeric form 
# prediction = model.predict(sample_numeric)
# print(prediction)

# ValueError: dtype='numeric' is not compatible with arrays of bytes/strings.Convert your data to numeric values explicitly instead.

from flask import Flask,request,jsonify
from prometheus_flask_exporter import PrometheusMetrics # PrometheusMetrics is a class

app = Flask(__name__)  # Creaed the flask object and stored in app and created a flask application

# BY default , route is always GET


@app.route("/") # GET request
def home():
    return jsonify(
        {
            "message" : "App is running."
        }
    )


@app.route("/predict",methods=["POST"]) # It is POST because it is requesting something 
def predict():
    data = request.get_json() # request is the module which have method get_json to get input from json and convert it into dict

    area = data["Area"] 
    bedroom = data["Bedrooms"]
    bathroom = data["Bathrooms"]
    floor = data["Floors"]
    year_built = data["YearBuilt"]
    loc = data["Location"]
    cond = data["Condition"]
    garage  = data["Garage"]

    prediction = model.predict([[area,bedroom,bathroom,floor,year_built,loc,cond,garage]])  # Here [[]] is the explict conversion to dataframe

    return jsonify({"prediction": prediction[0]})  # Here prediction will be get in form of array so to get value we need to put index


if __name__ =="__main__":
    app.run(host="0.0.0.0",port=5000)
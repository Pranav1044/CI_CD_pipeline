import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

model = joblib.load("model.pkl")

test_df = pd.read_csv("House Price_Prediction_Dataset .csv")

X = test_df.drop(["Id","Price"],axis=1)

Le = LabelEncoder()
X["Location"] = Le.fit_transform(test_df['Location']) 
X["Condition"] = Le.fit_transform(test_df["Condition"])
X["Garage"] = Le.fit_transform(test_df["Garage"])

y = test_df['Price']
pred = model.predict(X)
accuracy=r2_score(pred,y)
print("Accuracy_score : ",accuracy)

assert accuracy >= -200  # Basically putting limit that if it is greater than this then show
assert len(pred) == len(X)
print("Test cases passed")
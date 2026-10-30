# Here model will be trained and saved 



import pandas as pd
import joblib  # similar as pickle but in this file handling is not required.
df = pd.read_csv("House Price_Prediction_Dataset .csv")

# print(df.corr(numeric_only=True))
X = df.drop(["Id","Price"],axis=1)
Y = df["Price"]

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
X["Location"] = Le.fit_transform(df['Location']) 
X["Condition"] = Le.fit_transform(df["Condition"])
X["Garage"] = Le.fit_transform(df["Garage"])


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=66)
# print(X_train.info())
# print(X_train.shape)
# print(X.head())

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
# print(Y_pred)

from sklearn.metrics import r2_score
ac = r2_score(Y_test,Y_pred)
print("Accuracy : ",ac)

# Model save
joblib.dump(model,"model.pkl")
# in pickle , we have to do file handling to save the model but not in joblib
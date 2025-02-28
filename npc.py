import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


st.title("Experiance Vs Salary Prediction")
st.write("This model will predict saalry of the employes on the basis of their work experiance")

X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)

y=np.array([30000,35000,50000,70000,90000,130000,180000,250000,350000,500000]).reshape(-1,1)

sc_X=StandardScaler()
sc_y=StandardScaler()
X_scaled=sc_X.fit_transform(X)
y_scaled=sc_y.fit_transform(y)

svr_model=SVR(kernel='rbf',C=1000,epsilon=0.1)
svr_model.fit(X_scaled,y_scaled.ravel())

input=st.number_input("Enter the Year of Experiance of the employee:")
if input==0:
    st.write(f"Predicted Salary of{input}years of experiance:$3000")
else:
    predicted_salary=sc_y.inverse_transform(svr_model.predict(sc_X.transform([[input]])).reshape(-1,1))
    st.write(f"Predicted Salary for {input}years of experiance:${predicted_salary[0][0]:,.2f}")

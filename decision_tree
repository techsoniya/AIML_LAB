import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
dataset=pd.read_csv('PlayTennis.csv')
features=['Outlook','Temperature','Humidity','Wind']
X=dataset[features]
Y=dataset.PlayTennis

encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')
X_encoded=pd.DataFrame(encoder.fit_transform(X))
columns=encoder.get_feature_names(features)
X_train,X_test,y_train,y_test=train_test_split(X_encoded,Y,test_size=0.30,random_state=100)


dtree=DecisionTreeClassifier(criterion="entropy",random_state=100)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)

def classify_new_instance(outlook,temperature,humidity,wind,encoder):
    instance=[[outlook,temperature,humidity,wind]]
    instance_df=pd.DataFrame(instance,columns=features)
    instance_encoded=encoder.transform(instance_df)
    feature_names=encoder.get_feature_names(features)
    instance_encoded_df=pd.DataFrame(instance_encoded,columns=feature_names)
    prediction=dtree.predict(instance_encoded_df)
    return prediction[0]

pred=classify_new_instance("Rain","Mild","High","Strong",encoder)
print("Prediction:",pred)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

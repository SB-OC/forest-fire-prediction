import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import numpy as np
import joblib
import mlflow


df = pd.read_csv('/mnt/dataset/Algerian_forest_fires_dataset_CLEANED_NEW.csv')

# df.columns
# df.drop(['day','month','year'], axis=1, inplace=True)

df['Classes']= np.where(df['Classes']== 'not fire',0,1)
X = df.drop('FWI',axis=1)
y= df['FWI']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape, X_test.shape
X_train.corr()

# drop features which has correlation more than 0.75
# corr_features = correlation(X_train, 0.75)
# X_train.drop(corr_features,axis=1, inplace=True)
# X_test.drop(corr_features,axis=1, inplace=True)

X_train.shape, X_test.shape




#model creation


knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
mae = mean_absolute_error(y_test, knn_pred)
r2 = r2_score(y_test, knn_pred)


print("K_Neighbours Regressor")
print ("R2 Score value: {:.4f}".format(r2))
print ("MAE value: {:.4f}".format(mae))



mlflow.log_metric("explained_variance",r2)
mlflow.log_metric("mae",mae)


joblib.dump(lreg, "/mnt/result/model.joblib")

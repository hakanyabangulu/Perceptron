import pandas as pd
import numpy as np

train = pd.read_excel("DataForPerceptron.xlsx", sheet_name="TRAINData")
test  = pd.read_excel("DataForPerceptron.xlsx", sheet_name="TESTData")

X_train = train.iloc[:, 1:-1].values.astype(float) # All rows & All columns except the last columns & ID(First) columns. X's.
mean = X_train.mean(axis=0) # Columns mean for scaling.
std  = X_train.std(axis=0) # Columns std for scaling.
std[std == 0] = 1
X_scale = (X_train - mean) / std # Scale x.

y_train = train.iloc[:, -1].values # All rows & Last columns. => Y's.
vals = np.unique(y_train) # [2,4]
y_train = np.where(y_train == vals[0], -1, +1) # if 2 ; then -1 , else 1.


X_test = test.iloc[:, 1:1+X_train.shape[1]].values.astype(float) # Equal rows for test.
X_test_scale  = (X_test  - mean) / std # Scale test x.

def perceptron(X, y, epoch=1000): # Learning alg.
    n_row, n_column = X.shape
    w = np.zeros(n_column) # İnitial w = 0,0,0,... Length of the features.
    b = 0.0 # Bias.
    for _ in range(epoch):
        for i in range(n_row):
            z = np.dot(X[i], w) + b # z = w.x + b
            pred = 1 if z >= 0 else -1
            if pred != y[i]:
                w += y[i] * X[i]  # Updating w.
                b += y[i] # Updating b.
    return w,b

w,b = perceptron(X_scale, y_train) # Last w calculation.
z = np.dot(X_test_scale, w) + b # z = w.x + b
pred = np.where(z >= 0, 1, -1) # Prediction.

# Convert -1,1 to 2,4.
reverse = {-1: vals[0], 1: vals[1]} # if -1 ; then 2 , else 4.
out = np.array([reverse[p] for p in pred]) # List of preds.
ID = test.iloc[:, 0].values  # ID's.

# CSV olarak kaydet
out = pd.DataFrame({
    "ID ": ID,
    "Prediction": out
})
out.to_csv("Results.csv", index=False)
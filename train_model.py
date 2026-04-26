import csv
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = []
labels = []

for file in os.listdir("data"):
    if file.endswith(".csv"):
        with open(f"data/{file}", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                labels.append(row[0])
                data.append([float(x) for x in row[1:]])

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
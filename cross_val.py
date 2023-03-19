import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load the labeled accelerometer data from a CSV file
data = pd.read_csv("demo5.csv")

# Separate the features (accelerometer data) from the labels (posture)
X = data.drop("possture", axis=1)
y = data["possture"]

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Perform 5-fold cross-validation and compute the mean accuracy for each fold
scores = cross_val_score(dtc, X, y, cv=5)
mean_scores = np.mean(scores, axis=1)

# Plot the mean accuracy for each fold
fig, ax = plt.subplots()
ax.plot(range(1, 6), mean_scores, marker='o')
ax.set_xlabel('Fold')
ax.set_ylabel('Mean accuracy')
ax.set_title('Cross-validation scores')

# Train the decision tree classifier on all the data
dtc.fit(X, y)

# Plot the decision tree
from sklearn.tree import plot_tree
fig, ax = plt.subplots()
plot_tree(dtc, ax=ax)
plt.show()

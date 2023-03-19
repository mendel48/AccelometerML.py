import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('demo5.csv')

# Traitement des données
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])
y = df.iloc[:, -1]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Définition des modèles
dtc = DecisionTreeClassifier()
svm = SVC()
knn = KNeighborsClassifier()

# Apprentissage et évaluation des modèles
models = [dtc, svm, knn]
model_names = ['Decision Tree', 'SVM', 'KNN']
accuracies = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Sélection du meilleur modèle
best_model_index = accuracies.index(max(accuracies))
best_model_name = model_names[best_model_index]
print('Best Model:', best_model_name)

# Tracer un graphique pour visualiser les performances des différents modèles
plt.bar(model_names, accuracies)
plt.ylim([0, 1])
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

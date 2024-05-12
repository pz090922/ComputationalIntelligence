import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

def evaluate_model(model, predictions):
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



# Usunięcie zbędnych kolumn, takich jak matchId
data = pd.read_csv("match_data_v5.csv")

# Podział danych na cechy (X) i etykiety (y)
X = data.drop(columns=["matchId", "blueWin"])
y = data["blueWin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
# rf_predictions = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# evaluate_model(rf_model, rf_predictions)
# conf_matrix = confusion_matrix(y_test, rf_predictions)
# print(conf_matrix)
"""
Random Forest Classifier: 0.7415892672858617
[[1852  617]
 [ 635 1741]]
"""


# Support Vector Machine
# svm_model = SVC(random_state=42)
# svm_model.fit(X_train, y_train)
# svm_predictions = svm_model.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_predictions)
# evaluate_model(svm_model, svm_predictions)
# conf_matrix = confusion_matrix(y_test, svm_predictions)
# print(conf_matrix)
"""
Support Vector Machine Accuracy: 0.7420020639834881
[[1850  619]
 [ 631 1745]]
"""

# KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=11)
# knn.fit(X_train, y_train)
# pred_knn = knn.predict(X_test)
# accuracy_knn = accuracy_score(y_test, pred_knn)
# conf_matrix_knn = confusion_matrix(y_test, pred_knn)
# print(accuracy_knn,conf_matrix_knn)
""""
KNeighboursClassifier(3) 0.6850361197110423
Macierz błedu
[[1707  762]
 [ 764 1612]]
 
Dla 5
0.7011351909184727 
[[1753  716]
[ 732 1644]]
 
Dla 11
0.7242518059855522 
[[1804  665]
[ 671 1705]]
"""

mlp_model = MLPClassifier(hidden_layer_sizes=(6,3), activation='tanh', random_state=42)
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
conf_matrix = confusion_matrix(y_test, mlp_predictions)
print("Mean Cross-Validation Accuracy:", mlp_accuracy, conf_matrix)
"""
0.737874097007224 
[[1688  781]
 [ 489 1887]]
"""

"""
Im wyższa wartość AUC, tym lepiej model radzi sobie z rozróżnianiem między klasami. 
W przypadku AUC, wartość 0.5 oznacza model losowy, a wartość 1.0 oznacza idealny model.

CZUŁOŚĆ wykrywalnosc odpowiedzi TAK
"""
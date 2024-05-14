import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, recall_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import tensorflow as tf
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.special import softmax
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# https://www.kaggle.com/code/kirklin/game-winner-prediction-best-76-9-w-eda-finetune#Import-Data

def find_bestKfeatures(model, score_func=f_classif):
    # Select best k features
    k = -1
    max_score = 0
    for i in range(1, 39):
        selector = SelectKBest(score_func=score_func, k=i)
        pipeline = Pipeline([('selector', selector), ('model', model)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print("K: {}, score: {}".format(i, score))
        if score > max_score:
            k = i
            max_score = score
            selected_features_indices = selector.get_support(indices=True)
    print("The best K number: {}, score: {}".format(k, max_score))
    # return feature index list
    return list(selected_features_indices)


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


data = pd.read_csv("match_data_v5.csv")
remove1 = data[data['blueTeamTurretPlatesDestroyed'] > 15].index
data.drop(remove1, inplace=True)
remove2 = data[data['redTeamTurretPlatesDestroyed'] > 15].index
data.drop(remove2, inplace=True)
print(data.shape)
X = data.drop(columns=["matchId", "blueWin"])
y = data["blueWin"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# sns.set_context("notebook", font_scale=1.5)
# sns.pairplot(data=data, vars=('blueTeamControlWardsPlaced','blueTeamWardsPlaced'), hue='blueWin')
# plt.show()
# sns.pairplot(data=data, vars=('blueTeamTotalKills','blueTeamDragonKills','blueTeamHeraldKills'), hue='blueWin')
# plt.show()
# sns.pairplot(data=data, vars=('blueTeamTowersDestroyed','blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed'), hue='blueWin')
# plt.show()
# sns.pairplot(data=data, vars=('blueTeamTotalGold','blueTeamXp','blueTeamTotalDamageToChamps'), hue='blueWin')
# plt.show()
# Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
# rf_predictions = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
# evaluate_model(rf_model, rf_probabilities)
# conf_matrix = confusion_matrix(y_test, rf_predictions)
# rf_score = recall_score(y_test, rf_predictions)
# print(conf_matrix, rf_score, rf_accuracy)
"""
Random Forest Classifier: 0.7531806615776081
area: 0.84
recall_score: 0.755330377876293
[[1852  617]
 [ 635 1741]]
"""

# Support Vector Machine
# svm_model = SVC(random_state=42)
# svm_model.fit(X_train, y_train)
# svm_predictions = svm_model.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_predictions)
# svm_probabilities = svm_model.decision_function(X_test)
# evaluate_model(svm_model, svm_probabilities)
# conf_matrix = confusion_matrix(y_test, svm_predictions)
# svm_recall = recall_score(y_test, svm_predictions)
# print(conf_matrix, svm_recall, svm_accuracy)
"""
Support Vector Machine: 0.7517416086130463
area: 0.74
recall_score 7451229855810009
[[1850  619]
 [ 631 1745]]
"""

# KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=11)
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)
# knn_accuracy = accuracy_score(y_test, knn_pred)
# knn_conf_matrix = confusion_matrix(y_test, knn_pred)
# knn_distances, knn_indices = knn.kneighbors(X_test)
# knn_probabilities = 1 - knn_distances / np.max(knn_distances)
# knn_probabilities = softmax(knn_probabilities, axis=1)
# evaluate_model(knn, knn_probabilities[:, 1])
# knn_recall = recall_score(y_test, knn_pred)
# print(knn_conf_matrix, knn_recall, knn_accuracy)
""""
KNeighboursClassifier(3): 0.6850361197110423
[[1707  762]
 [ 764 1612]]
 
Dla 5
0.7011351909184727 
[[1753  716]
[ 732 1644]]
 
Dla 11
0.7337977622968124
area: 0.72
recall_score: 0.7251908396946565
[[1766  613]
 [ 648 1710]]
"""

# mlp_model = MLPClassifier(hidden_layer_sizes=(6,3), activation='relu', random_state=42) # relu chyba najlepesze
# mlp_model.fit(X_train, y_train)
# mlp_pred = mlp_model.predict(X_test)
# mlp_accuracy = accuracy_score(y_test, mlp_pred)
# conf_matrix = confusion_matrix(y_test, mlp_pred)
# mlp_probabilities = mlp_model.predict_proba(X_test)[:, 1]
# evaluate_model(mlp_model, mlp_probabilities)
# mlp_recall = recall_score(y_test, mlp_pred)
# print(mlp_accuracy, mlp_recall ,conf_matrix)
"""
0.737874097007224 
area: 0.74
recall_score: 0.7941919191919192
[[1688  781]
 [ 489 1887]]
"""

# nb = GaussianNB()
# nb.fit(X_train, y_train)
# pred_nb = nb.predict(X_test)
# accuracy_nb = accuracy_score(y_test, pred_nb)
# conf_matrix_nb = confusion_matrix(y_test, pred_nb)
# evaluate_model(nb, pred_nb)
# recall_nb = recall_score(y_test, pred_nb)
# print(accuracy_nb, recall_nb ,conf_matrix_nb)
"""
0.7415892672858617 
area: 0.74
recall_score: 0.7310606060606061
[[1856  613]
 [ 639 1737]]
"""

# model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         tf.keras.layers.Dropout(0.5),  # Dodanie warstwy Dropout
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#         tf.keras.layers.Dropout(0.5),  # Dodanie warstwy Dropout
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train_normalized, y_train, epochs=50, batch_size=32,
#                     validation_data=(X_test_normalized, y_test),
#                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
# loss, accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
# predictions = model.predict(X_test_normalized)
# threshold = 0.5
#
# binary_predictions = (predictions > threshold).astype(int)
#
# conf_matrix_m = confusion_matrix(y_test, binary_predictions)
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
# evaluate_model(history, predictions)
# print("Model Accuracy:", conf_matrix_m)
"""
0.7415892481803894
Model Accuracy: 0.7574414014816284
area: 0.85
[[1811  568]
 [ 593 1765]]
"""

best_log_reg_model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=400, C=0.1)
# lr_features = find_bestKfeatures(model_lr, score_func=f_classif)
best_log_reg_model.fit(X_train_normalized, y_train)
y_pred = best_log_reg_model.predict(X_test_normalized)
y_prob = best_log_reg_model.predict_proba(X_test_normalized)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
evaluate_model(best_log_reg_model, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
recall_log_reg = recall_score(y_test, y_pred)
print(accuracy, recall_log_reg, conf_matrix)
print(classification_report(y_test, y_pred))
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
specificity = TN / (TN + FP)
print(specificity)

"""
Accuracy: 0.7584969389909225
Recall: 0.7502120441051738
[[1824  555]
 [ 589 1769]]
 Swoistość 0.766708
"""
binarizer = Binarizer(threshold=0)
X_train_bin = binarizer.fit_transform(X_train_normalized)
X_test_bin = binarizer.transform(X_test_normalized)
frequent_itemsets = apriori(pd.DataFrame(X_train_bin, columns=X_train.columns), min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(rules)
rule_with_max_zhangs_metric = rules[rules['lift'] == rules['lift'].min()]

print("Rule with max Zhang's metric:")
for index, row in rule_with_max_zhangs_metric.iterrows():
    print("Antecedents:", row['antecedents'])
    print("Consequents:", row['consequents'])
    print("Antecedent support:", row['antecedent support'])
    print("Consequent support:", row['consequent support'])
    print("Support:", row['support'])
    print("Confidence:", row['confidence'])
    print("Lift:", row['lift'])
    print("Leverage:", row['leverage'])
    print("Conviction:", row['conviction'])
    print("Zhang's metric:", row['zhangs_metric'])
    print("\n")
"""
Im wyższa wartość AUC, tym lepiej model radzi sobie z rozróżnianiem między klasami. 
W przypadku AUC, wartość 0.5 oznacza model losowy, a wartość 1.0 oznacza idealny model.

CZUŁOŚĆ wykrywalnosc odpowiedzi TAK


Dodano warstwy Dropout w modelu, aby losowo wyłączać część neuronów w trakcie treningu i zapobiegać nadmiernemu dopasowaniu.
Zastosowano regularyzację L2, aby kontrolować wielkość wag i zapobiec nadmiernemu wzrostowi.
Dodano wczesne zatrzymywanie (EarlyStopping), aby monitorować proces uczenia się i zatrzymać trening, gdy nie następuje poprawa wydajności na danych walidacyjnych.
"""

"""
antecedents: To zestaw cech (lub pojedyncza cecha), które występują w poprzednim elemencie reguły.

consequents: To zestaw cech (lub pojedyncza cecha), które występują w następnym elemencie reguły.

support: Określa, jak często zestaw cech (antecedents lub consequents) pojawia się razem w całym zbiorze danych. 
Wartość support dla konkretnej reguły asocjacyjnej mówi nam, jak często ta reguła występuje w danych. Wartość ta jest wyrażana 
jako prawdopodobieństwo wystąpienia danego zestawu cech w zbiorze danych.

confidence: Określa prawdopodobieństwo, że zestaw cech w consequents wystąpi, pod warunkiem, że zestaw cech w 
antecedents wystąpił. Wartość confidence dla konkretnej reguły asocjacyjnej mówi nam, jak często cechy z consequents 
występują w danych, gdy cechy z antecedents występują.

lift: Określa siłę zależności między zestawami cech antecedents i consequents. Wartość lift większa niż 1 
oznacza, że występuje dodatnia zależność między zestawami cech, podczas gdy wartość mniejsza niż 1 oznacza, że 
występuje negatywna zależność, a wartość równa 1 oznacza brak zależności.

zhangs_metric: Jest to metryka, która ocenia jakość reguł asocjacyjnych, uwzględniając ich support, 
confidence i lift. Im wyższa wartość tej metryki, tym bardziej wartościowa jest reguła.

Wartości w kolumnach antecedents i consequents są zestawami cech, a wartości w pozostałych kolumnach są 
odpowiednio miarami oceny tych reguł asocjacyjnych. Analizując te wyniki, możemy zidentyfikować istotne zależności 
między cechami w danych, co może prowadzić do ciekawych wniosków lub strategii w kontekście analizy danych.


Poprawa wydajności algorytmów: Niektóre algorytmy uczące, takie jak regresja logistyczna czy metody oparte na odległościach (np. k-NN), 
mogą działać lepiej, gdy cechy są ustandaryzowane.

Unikanie wpływu wartości odstających: Skalowanie danych pomaga w minimalizacji wpływu wartości odstających na model, 
ponieważ ogranicza wpływ ekstremalnych wartości na proces uczenia.
"""

"""
Antecedents: frozenset({'blueTeamTotalDamageToChamps', 'redTeamTotalGold', 'blueTeamTotalKills'})
Consequents: frozenset({'blueTeamTotalGold', 'redTeamTotalKills', 'redTeamTotalDamageToChamps'})
Confidence: 0.6441541892322395
Lift: 3.9474235373779027
Conviction: 2.351626814866641
Zhang's metric: 0.8949273845752262
"""

"""
Masz rację, przepraszam za nieprecyzyjne stwierdzenie. Lift wynoszący 1 i niska wartość Zhang's metric sugerują, że nie ma silnej zależności między przyczynami a skutkami w tej konkretnej regule asocjacyjnej. Oznacza to, że występowanie wardów kontrolnych przez obie drużyny niekoniecznie prowadzi do zniszczenia wież i zabicia smoka przez drużynę czerwoną.

Wartości te wskazują, że nawet jeśli występują pewne powiązania między wardami kontrolnymi a zdarzeniami w grze, są one słabe lub przypadkowe w kontekście tej konkretnej reguły. Dlatego faktycznie możemy stwierdzić, że zależność między wardami kontrolnymi a zniszczeniem wież i zabiciem smoka jest bardzo mała lub nieistniejąca w analizowanym zestawie danych.
"""


"""
redTeamTurretsPlatesDestroyed to znaczy ile oni stracili plateów

"""

"""
Antecedents: frozenset({'redTeamControlWardsPlaced', 'blueTeamControlWardsPlaced'})
Consequents: frozenset({'redTeamDragonKills', 'redTeamTowersDestroyed'})
Confidence: 0.43046650453239005
Lift: 1.00042675430881
Conviction: 1.0003224131228063
Zhang's metric: 0.0005603252214893278
"""
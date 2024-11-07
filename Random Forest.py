import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# dataset
percorso_dataset = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\2. Online Gaming Latency - OneHotEncoder.csv"
df = pd.read_csv(percorso_dataset)


# conversione colonna "date" in datetime
df['Date'] = pd.to_datetime(df['Date'])

# estrazione giorno  mese anno dalla colonna "Date"
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


# rimoz colonna "date" originale
df = df.drop(columns=['Date'])

# rimozione righe con nan
df = df.dropna()

# calcolo soglia per definire la colonna "classRTT"
rtt_media = df['Cur. Round Trip Latency Median'].mean()

#fefinizione colonna "classRTT" basata sulla soglia
df['classRTT'] = np.where(df['Cur. Round Trip Latency Median'] >= rtt_media, 'cattivo', 'buono')

# fefinizione variabili indip e dipendenti
variabili_indipendenti = ['SS-RSRP', 'SS-RSRQ', 'SS-SINR', 'RSRP', 'RSRQ', 'SINR',
                           'UE Mode_5G-enabled',
                          'Scenario_OD', 'Scenario_OW', 'Operator_Op2',
]
variabile_dipendente = 'classRTT'

# conversione delle etichette in valori binari
label_encoder = LabelEncoder()
df['classRTT'] = label_encoder.fit_transform(df['classRTT'])

# set di addestramento e test
X = df[variabili_indipendenti]
y = df[variabile_dipendente]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# dtandardizzaz delle variabili indipendenti
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# griglia di iperparametri per la grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

#creazione del modello Random Forest
modelloRF = RandomForestClassifier(random_state=42)

# configurazione della grid search con validazione incrociata
grid_search = GridSearchCV(estimator=modelloRF, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


# esecuzione della Grid Search
grid_search.fit(X_train_scaled, y_train)

# migliori parametri trovati 
best_params = grid_search.best_params_
print("Migliori parametri trovati:", best_params)

#migliore modello trovato 
best_model = grid_search.best_estimator_


# previsioni usando il miglior modello
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

#calcolo accuracy
train_accuratezza = accuracy_score(y_train, y_pred_train)
test_accuratezza = accuracy_score(y_test, y_pred_test)

print("Accuratezza sul set di addestramento:", train_accuratezza)
print("Accuratezza sul set di test:", test_accuratezza)



# Visualizzaz del report di classificazione
print("\nReport di classificazione sul set di test:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# calcolo curva ROC e AUC
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilità della classe positiva
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

# visualizzazione della curva roc
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, 0, tpr, color='#FFFFCC')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
plt.xlim([min(fpr), 1.0])  # Adjusted x-axis to start from the first value
plt.ylim([min(tpr), 1.05])  # Adjusted y-axis to start from the first value
plt.xlabel('Tasso di falsi positivi')
plt.ylabel('Tasso di positività reale')
plt.title('Curva ROC(Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# calcolo della curva precision-recall e AUC
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print(f"AUC Precision-Recall: {pr_auc:.4f}")

#visualizzazione della curva Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.2f}')
plt.fill_between(recall, 0, precision, color='#FFFFCC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend(loc="lower left")
plt.xlim([min(recall), 1.0])  # Adjusted x-axis to start from the first value
plt.ylim([min(precision), 1.05])  # Adjusted y-axis to start from the first value
plt.grid(True)
plt.show()

#curva apprendimento
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=5, n_jobs=-1, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

# media e deviazione standard dei punteggi
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#visualizzazione curva di apprendimento
plt.figure(figsize=(12, 8))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Punteggio di addestramento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Punteggio di cross-validation')
plt.xlabel('Numero di campioni di addestramento')
plt.ylabel('Punteggio')
plt.title('Curva di Apprendimento')
plt.legend(loc='best')
plt.grid()
plt.show()


#feature imporatnce
feature_importances = best_model.feature_importances_
features_df = pd.DataFrame({'Feature': variabili_indipendenti, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

# arrotonda feature imp a due decimali e moltiplica x100 per ottenere percentuali
features_df['Importance'] = (features_df['Importance'] * 100).round(0).astype(int)

#stampa il df con feature imp
print("\nImportanza delle feature:")
print(features_df)


#visualizzazione tabella
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=features_df.values, colLabels=features_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Importanza delle variabili', fontweight='bold')
plt.show()

#visualizzazione delle feat imp
plt.figure(figsize=(12, 8))
plt.barh(features_df['Feature'], features_df['Importance'], color='blue')
plt.xlabel('Importanza')
plt.ylabel('Caratteristica')
plt.title('Importanza delle Caratteristiche')
plt.gca().invert_yaxis()
plt.show()

# N campioni per variabile
sample_counts = df[variabili_indipendenti].count()
print("\nNumero di campioni per ogni variabile:")
print(sample_counts)

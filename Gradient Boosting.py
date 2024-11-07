import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

#dataset
percorso_dataset = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\2. Online Gaming Latency - OneHotEncoder.csv"
df = pd.read_csv(percorso_dataset)

# conversione di "date" in datetime
df['Date'] = pd.to_datetime(df['Date'])

#Estrazione di giorno, mese e anno dalla colonna "date"
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


#Rimozione della colonna 'Date' originale
df = df.drop(columns=['Date'])

#rimozione delle righe con  na 
df = df.dropna()

# calcolo della soglia per definire la colonna 'classRTT'
rtt_media = df['Cur. Round Trip Latency Median'].mean()

# definizione dellla colonna "classRTT" basata sulla soglia
df['classRTT'] = np.where(df['Cur. Round Trip Latency Median'] >= rtt_media, 'cattivo', 'buono')

# definizione di var indip e dipendente
variabili_indipendenti = ['SS-RSRP', 'SS-RSRQ', 'SS-SINR', 'RSRP', 'RSRQ', 'SINR',
                          'UE Mode_5G-enabled',
                          'Scenario_OD', 'Scenario_OW', 'Operator_Op2',
                         ]
variabile_dipendente = 'classRTT'

# conversione etichette in valori binari
label_encoder = LabelEncoder()
df['classRTT'] = label_encoder.fit_transform(df['classRTT'])

# Divisione del dataset in set di addestramento e test
X = df[variabili_indipendenti]
y = df[variabile_dipendente]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardizzazione variabili indipendenti
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# definizione griglia iperparametri 
param_grid_gbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#creazione modello Gradient Boosting
modelloGB = GradientBoostingClassifier(random_state=42)

# configurazione della randomized search con validazione incrociata
random_search = RandomizedSearchCV(estimator=modelloGB, param_distributions=param_grid_gbm, n_iter=100,
                                   cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=2)

#esecuzione randomized search
random_search.fit(X_train_scaled, y_train)

# parametri migliori trovati dalla randomized search
best_params = random_search.best_params_
print("Migliori parametri trovati:", best_params)


#migliore modello trovato dalla Randomized Search
best_model = random_search.best_estimator_

# previsioni utilizzando il modello migliore
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)


#calcolo accuratezza
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Accuratezza sul set di addestramento:", train_accuracy)
print("Accuratezza sul set di test:", test_accuracy)


# visualizzazione report classificazione
print("\nReport di classificazione sul set di test:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

#calcolo curva ROC e AUC
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilità della classe positiva
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

# visualizzazione curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, 0, tpr, color='#FFFFCC')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di falsi positivi')
plt.ylabel('Tasso di positività reale')
plt.title('Curva ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#calcolo curva Precision-Recall e AUC
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print(f"AUC Precision-Recall: {pr_auc:.4f}")

#visualizzazione curva precision-recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.2f}')
plt.fill_between(recall, 0, precision, color='#FFFFCC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend(loc="lower left")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.show()


# Curva apprendimento
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=5, n_jobs=-1, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

#media e deviazione standard di punteggi
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# visualizzazione curva di apprendimento
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

# feature importance
feature_importances = best_model.feature_importances_
features_df = pd.DataFrame({'Feature': variabili_indipendenti, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

# arrotonda feature importance a due decimali e moltiplica x 100 per ottenere percentuali
features_df['Importance'] = (features_df['Importance'] * 100).round(0).astype(int)

# stampa dataframe con feature importances
print("\nImportanza delle feature:")
print(features_df)

# visualizzazione delle feature importance
plt.figure(figsize=(12, 8))
plt.barh(features_df['Feature'], features_df['Importance'], color='blue')
plt.xlabel('Importanza')
plt.ylabel('Caratteristica')
plt.title('Importanza delle Caratteristiche')
plt.gca().invert_yaxis()
plt.show()


# numero campioni per variabile
sample_counts = df[variabili_indipendenti].count()
print("\nNumero di campioni per ogni variabile:")
print(sample_counts)

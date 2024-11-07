import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# dataset
percorso_dataset = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\1.Online Gaming Latency - PreProcessed.csv"

# Seleziona le colonne principali
colonneselezionate = ["Date", "Time", "SS-RSRP", "SS-RSRQ", "SS-SINR", "RSRP", "RSRQ", "SINR", 
                      "Cur. Round Trip Latency Median", "Cur. Channel QoS 3GPP", 'LTE CA',
                      'UE Mode', 'Scenario', 'Operator']

#carica dataset
df = pd.read_csv(percorso_dataset, usecols=colonneselezionate)

#converti i valori della colonna 'Cur. Round Trip Latency Median' da secondi a millisecondi
df['Cur. Round Trip Latency Median'] *= 1000

# conta i valori mancanti prima di eliminarli
valori_mancanti_prima = df.isnull().sum()
print("Numero di valori mancanti prima dell'eliminazione:")
print(valori_mancanti_prima)

# drop righe con valori mancanti
df.dropna(inplace=True)

#conta i valori mancanti dopo averli eliminati
valori_mancanti_dopo = df.isnull().sum()
print("\nNumero di valori mancanti dopo l'eliminazione:")
print(valori_mancanti_dopo)



# one-hot-encoder varibili categoriali con drop first
encoder = OneHotEncoder(drop='first')
categoriali = ['LTE CA', 'UE Mode', 'Scenario', 'Operator']
encoded_data = encoder.fit_transform(df[categoriali])


# converte la matrice codificata in un df pandas
one_hot_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categoriali))

# unisce il df base con quello one-hot encod
df_encoded = pd.concat([df.drop(columns=categoriali), one_hot_encoded], axis=1)


# salva il nuovo df in  CSV
Dataset_Latency_Pulito_ONEHOT2 = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\2. Online Gaming Latency - OneHotEncoder.csv"
df_encoded.to_csv(Dataset_Latency_Pulito_ONEHOT2, index=False)

print("\nNuovo dataset salvato con successo in:", )

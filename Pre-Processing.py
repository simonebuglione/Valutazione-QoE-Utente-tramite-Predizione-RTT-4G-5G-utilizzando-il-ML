import pandas as pd

#dataset 
dataset_path = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\Latency Tests - Online Gaming - Active Measurements.csv"

data = pd.read_csv(dataset_path, na_values='?')

# converti i valori da secondi a millisecondi
data['Cur. Round Trip Latency Median'] *= 1000

# raggruppa per campagna e applica FFILL alle colonne
data_gruppo = data.groupby('campaign').apply(lambda group: group.ffill())

# percorso di salvataggio del dataset buovo
output_path = r"C:\Users\bugli\Desktop\Online Gaming Latency\Dataset\1.Online Gaming Latency - PreProcessed.csv"

#salva il nuovo dataset nella cartella 
data_gruppo.to_csv(output_path, index=False)

print(f"Il dataset è stato salvato in: {output_path}")
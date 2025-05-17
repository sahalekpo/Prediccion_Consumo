import pandas as pd
import numpy as np

np.random.seed(12)  # Para reproducibilidad

n = 10000


temperatura = np.random.normal(loc=22, scale=5, size=n)
personas = np.random.randint(1, 6, size=n).astype(float)  # Convertido a float
electrodomesticos = np.random.randint(5, 21, size=n).astype(float)  # Convertido a float


consumo = (temperatura * 0.8 + personas * 3 + electrodomesticos * 1.5 + np.random.normal(0, 3, n))


for col in [temperatura, personas, electrodomesticos, consumo]:
    idx_nan = np.random.choice(n, size=int(n * 0.05), replace=False)
    col[idx_nan] = np.nan


outlier_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
consumo[outlier_idx] += np.random.normal(100, 20, size=len(outlier_idx))


df = pd.DataFrame({
    'Temperatura': temperatura,
    'Personas': personas,
    'Electrodomesticos': electrodomesticos,
    'Consumo_kWh': consumo
})


df.to_csv('consumo_hogar.csv', index=False)
print("Archivo 'consumo_hogar.csv' generado con éxito.")

# ==========================
# 📦 Importar librerías
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# ==========================
# 🪟 Ventana emergente
# ==========================
def mostrar_ventana_texto(titulo, contenido):
    ventana = tk.Tk()
    ventana.title(titulo)
    ventana.geometry("700x400")
    texto = ScrolledText(ventana, font=("Consolas", 10))
    texto.pack(fill="both", expand=True)
    texto.insert(tk.END, contenido)
    texto.configure(state='disabled')
    ventana.mainloop()

# ==========================
# 1️⃣ Cargar datos
# ==========================
df = pd.read_csv("consumo_hogar.csv")
df_original = df.copy()

descripcion = "🔍 INFORMACIÓN GENERAL\n" + str(df.info()) + "\n"
descripcion += "\n📊 ESTADÍSTICAS DESCRIPTIVAS\n" + str(df.describe()) + "\n"
descripcion += "\n❓ VALORES NULOS DETECTADOS\n" + str(df.isnull().sum())
mostrar_ventana_texto("Paso 1: Análisis Inicial", descripcion)

# ==========================
# 2️⃣ Violinplots de imputación de nulos
# ==========================
df_before = df_original.copy()
df_before['estado'] = 'Antes'
df_after = df.copy()
df_after.fillna(df_after.mean(), inplace=True)
df_after['estado'] = 'Después'

df_combined = pd.concat([df_before, df_after]).reset_index(drop=True)

columnas = ['Temperatura', 'Personas', 'Electrodomesticos', 'Consumo_kWh']

plt.figure(figsize=(12, 6))
for i, col in enumerate(columnas):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(data=df_combined, x='estado', y=col)
    plt.title(f"{col}: Antes vs Después de imputar nulos")
plt.tight_layout()
plt.show()

mensaje_nulos = "✅ Valores nulos imputados con la media. Comparación mostrada con violinplots para cada variable."
mostrar_ventana_texto("Paso 2: Imputación de Nulos", mensaje_nulos)

# ==========================
# 3️⃣ Boxplots por variable (sin transformación)
# ==========================
variables_estables = ['Temperatura', 'Personas', 'Electrodomesticos']
plt.figure(figsize=(12, 4))
for i, col in enumerate(variables_estables):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df_after[col])
    plt.title(f"{col} (sin transformación)")
plt.suptitle("Boxplots de variables sin transformación", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Boxplot de Consumo antes y después del log
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_after['Consumo_kWh'])
plt.title("Consumo_kWh (Antes de log)")
df_after['Consumo_kWh_log'] = np.log1p(df_after['Consumo_kWh'])
plt.subplot(1, 2, 2)
sns.boxplot(y=df_after['Consumo_kWh_log'])
plt.title("Consumo_kWh_log (Después de log)")
plt.suptitle("Boxplots de Consumo antes y después del log", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ==========================
# 4️⃣ Detección de outliers
# ==========================
z_scores = np.abs(stats.zscore(df_after['Consumo_kWh']))
df_after['es_outlier'] = z_scores > 3
mensaje_consumo = f"""
⚙️ TRANSFORMACIÓN LOG Y OUTLIERS

- Se aplicó log1p a 'Consumo_KWh'
- Se detectaron {df_after['es_outlier'].sum()} outliers mediante Z-score
- Los outliers NO fueron eliminados, solo marcados
"""
mostrar_ventana_texto("Paso 3: Transformación y Outliers", mensaje_consumo)

# ==========================
# 5️⃣ Visualización exploratoria
# ==========================
variables = ['Temperatura', 'Personas', 'Electrodomesticos']
plt.figure(figsize=(15, 4))
for i, var in enumerate(variables):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df_after, x=var, y='Consumo_kWh_log', alpha=0.5)
    plt.title(f'{var} vs Consumo_kWh_log')
plt.tight_layout()
plt.show()

# ==========================
# 6️⃣ Preparación del modelo
# ==========================
X = df_after[['Temperatura', 'Personas', 'Electrodomesticos']]
y_log = df_after['Consumo_kWh_log']
y_real = df_after['Consumo_kWh']
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, _, y_train_real, y_test_real = train_test_split(X, y_real, test_size=0.2, random_state=42)

# ==========================
# 7️⃣ Modelado
# ==========================
resultados = []
y_pred_poly_dict = {}

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train_log)
y_pred_log = modelo_lineal.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
mse_lineal = mean_squared_error(y_test_real, y_pred_real)
r2_lineal = r2_score(y_test_real, y_pred_real)
resultados.append({'Modelo': 'Lineal', 'MSE': mse_lineal, 'R2': r2_lineal})

for grado in [2, 3]:
    poly = PolynomialFeatures(degree=grado, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y_log, test_size=0.2, random_state=42)
    _, _, y_train_real_p, y_test_real_p = train_test_split(X_poly, y_real, test_size=0.2, random_state=42)

    modelo_poly = LinearRegression()
    modelo_poly.fit(X_train_p, y_train_p)
    y_pred_log_poly = modelo_poly.predict(X_test_p)
    y_pred_real_poly = np.expm1(y_pred_log_poly)

    mse = mean_squared_error(y_test_real_p, y_pred_real_poly)
    r2 = r2_score(y_test_real_p, y_pred_real_poly)

    y_pred_poly_dict[grado] = (y_test_real_p, y_pred_real_poly)
    resultados.append({
        'Modelo': f'Polinómico grado {grado}',
        'MSE': mse,
        'R2': r2
    })

# ==========================
# 8️⃣ Gráficas de predicción vs. real
# ==========================
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], '--r')
plt.xlabel("Consumo Real")
plt.ylabel("Consumo Predicho")
plt.title("Lineal: Real vs Predicho (escala original)")
plt.tight_layout()
plt.show()

for grado, (y_real_p, y_pred_p) in y_pred_poly_dict.items():
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_real_p, y=y_pred_p, alpha=0.5)
    plt.plot([y_real_p.min(), y_real_p.max()], [y_real_p.min(), y_real_p.max()], '--r')
    plt.xlabel("Consumo Real")
    plt.ylabel("Consumo Predicho")
    plt.title(f"Polinómico grado {grado}: Real vs Predicho")
    plt.tight_layout()
    plt.show()

# ==========================
# 9️⃣ Resumen final
# ==========================
resumen = "🧾 RESUMEN FINAL DEL MODELADO\n"
resumen += "="*45 + "\n"
resumen += "\n📊 MÉTRICAS DE LOS MODELOS (ESCALA ORIGINAL):\n"
for r in resultados:
    resumen += f" - {r['Modelo']}: MSE = {r['MSE']:.2f}, R² = {r['R2']:.4f}\n"
resumen += "\n✅ Proceso ejecutado correctamente."
mostrar_ventana_texto("Paso 4: Resultados Finales", resumen)

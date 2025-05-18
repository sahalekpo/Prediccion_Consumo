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

# ==========================
# 1️⃣ Cargar los datos
# ==========================
df = pd.read_csv("consumo_hogar.csv")

# ==========================
# 2️⃣ Análisis inicial
# ==========================
print("🔍 Información general:")
print(df.info())

print("\n📊 Estadísticas descriptivas:")
print(df.describe())

print("\n❓ Valores nulos por columna:")
print(df.isnull().sum())

# ==========================
# 3️⃣ Limpieza de datos
# ==========================

# Imputar valores nulos con la media
df.fillna(df.mean(), inplace=True)
print("\n✅ Valores nulos imputados con la media.")

# Guardar copia original para comparación si se desea después
df_original = df.copy()

print("\n❓ Valores nulos por columna:")
print(df.isnull().sum())

# Visualizar boxplots antes de detectar outliers
plt.figure(figsize=(12, 4))
for i, col in enumerate(df.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"{col} (Boxplot)")
plt.tight_layout()
plt.show()

# Detectar outliers en 'Consumo_kWh' con Z-score y marcar sin modificar
z_scores = np.abs(stats.zscore(df['Consumo_kWh']))
df['es_outlier'] = z_scores > 3
print(f"\n⚠️ Outliers detectados en 'Consumo_kWh': {df['es_outlier'].sum()}")

# ==========================
# 4️⃣ Transformación logarítmica
# ==========================
df['Consumo_kWh_log'] = np.log1p(df['Consumo_kWh'])

# ==========================
# 5️⃣ Visualización exploratoria
# ==========================
variables = ['Temperatura', 'Personas', 'Electrodomesticos']
plt.figure(figsize=(15, 4))
for i, var in enumerate(variables):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df, x=var, y='Consumo_kWh_log', alpha=0.5)
    plt.title(f'{var} vs Consumo_kWh_log')
    plt.xlabel(var)
    plt.ylabel("Consumo_kWh_log")
plt.tight_layout()
plt.show()

# ==========================
# 6️⃣ Preparar variables
# ==========================
X = df[['Temperatura', 'Personas', 'Electrodomesticos']]
y_log = df['Consumo_kWh_log']  # objetivo en escala log
y_real = df['Consumo_kWh']     # objetivo original para evaluación final

# Dividir para modelado
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, _, y_train_real, y_test_real = train_test_split(X, y_real, test_size=0.2, random_state=42)

# ==========================
# 7️⃣ Regresión lineal múltiple
# ==========================
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train_log)
y_pred_log = modelo_lineal.predict(X_test)
y_pred_real = np.expm1(y_pred_log)  # convertir a escala original

# Evaluar en escala original
mse_lineal = mean_squared_error(y_test_real, y_pred_real)
r2_lineal = r2_score(y_test_real, y_pred_real)

print("\n📈 Modelo de Regresión Lineal (con log transformado)")
for var, coef in zip(X.columns, modelo_lineal.coef_):
    print(f" - {var}: {coef:.4f}")
print(f"Intercepto: {modelo_lineal.intercept_:.4f}")
print(f"MSE (escala original): {mse_lineal:.2f}")
print(f"R²  (escala original): {r2_lineal:.4f}")

# ==========================
# 8️⃣ Regresión polinómica (grados 2 y 3)
# ==========================
resultados = []
y_pred_poly_dict = {}

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

    print(f"\n📈 Modelo Polinómico grado {grado} (con log transformado)")
    print(f"MSE (original): {mse:.2f}")
    print(f"R²  (original): {r2:.4f}")

# ==========================
# 9️⃣ Comparación de modelos
# ==========================
resultados.insert(0, {'Modelo': 'Lineal', 'MSE': mse_lineal, 'R2': r2_lineal})
df_resultados = pd.DataFrame(resultados)
print("\n📊 Comparación de Modelos (en escala original):")
print(df_resultados)

# ==========================
# 🔟 Gráficos Real vs Predicho
# ==========================

# Lineal
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], '--r')
plt.xlabel("Consumo Real")
plt.ylabel("Consumo Predicho")
plt.title("🔍 Regresión Lineal: Real vs Predicho")
plt.tight_layout()
plt.show()

# Polinómicos grado 2 y 3
for grado, (y_real_p, y_pred_p) in y_pred_poly_dict.items():
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_real_p, y=y_pred_p, alpha=0.5)
    plt.plot([y_real_p.min(), y_real_p.max()], [y_real_p.min(), y_real_p.max()], '--r')
    plt.xlabel("Consumo Real")
    plt.ylabel("Consumo Predicho")
    plt.title(f"📊 Polinómico grado {grado}: Real vs Predicho")
    plt.tight_layout()
    plt.show()

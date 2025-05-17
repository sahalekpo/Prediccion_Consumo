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

# Visualizar boxplots antes de limpieza completa
plt.figure(figsize=(12, 4))
for i, col in enumerate(df.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"{col} (antes)")
plt.tight_layout()
plt.show()

# Reemplazar outliers en Consumo_kWh con el promedio (no eliminar)
z_scores = np.abs(stats.zscore(df['Consumo_kWh']))
consumo_promedio = df['Consumo_kWh'][(z_scores < 3)].mean()
df.loc[z_scores >= 3, 'Consumo_kWh'] = consumo_promedio

# Visualizar boxplots después
plt.figure(figsize=(12, 4))
for i, col in enumerate(df.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"{col} (después)")
plt.tight_layout()
plt.show()

# Verificación
print(f"\n📦 DataFrame limpio: {df.shape[0]} registros")
print(df.isnull().sum())

# ==========================
# 4️⃣ Visualización exploratoria
# ==========================
variables = ['Temperatura', 'Personas', 'Electrodomesticos']
plt.figure(figsize=(15, 4))
for i, var in enumerate(variables):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df, x=var, y='Consumo_kWh', alpha=0.5)
    plt.title(f'{var} vs Consumo_kWh')
    plt.xlabel(var)
    plt.ylabel("Consumo_kWh")
plt.tight_layout()
plt.show()

# ==========================
# 5️⃣ Preparar variables
# ==========================
X = df[['Temperatura', 'Personas', 'Electrodomesticos']]
y = df['Consumo_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 6️⃣ Regresión lineal múltiple
# ==========================
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)
y_pred_lineal = modelo_lineal.predict(X_test)

mse_lineal = mean_squared_error(y_test, y_pred_lineal)
r2_lineal = r2_score(y_test, y_pred_lineal)

print("\n📈 Modelo de Regresión Lineal")
for var, coef in zip(X.columns, modelo_lineal.coef_):
    print(f" - {var}: {coef:.4f}")
print(f"Intercepto: {modelo_lineal.intercept_:.4f}")
print(f"MSE: {mse_lineal:.2f}")
print(f"R²: {r2_lineal:.4f}")

# ==========================
# 7️⃣ Regresión polinómica (grados 2 y 3)
# ==========================
resultados = []
y_pred_poly_dict = {}

for grado in [2, 3]:
    poly = PolynomialFeatures(degree=grado, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    modelo_poly = LinearRegression()
    modelo_poly.fit(X_train_p, y_train_p)
    y_pred_poly = modelo_poly.predict(X_test_p)

    mse = mean_squared_error(y_test_p, y_pred_poly)
    r2 = r2_score(y_test_p, y_pred_poly)

    y_pred_poly_dict[grado] = (y_test_p, y_pred_poly)

    resultados.append({
        'Modelo': f'Polinómico grado {grado}',
        'MSE': mse,
        'R2': r2
    })

    print(f"\n📈 Modelo Polinómico grado {grado}")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")

# ==========================
# 8️⃣ Comparación de modelos
# ==========================
resultados.insert(0, {'Modelo': 'Lineal', 'MSE': mse_lineal, 'R2': r2_lineal})
df_resultados = pd.DataFrame(resultados)
print("\n📊 Comparación de Modelos:")
print(df_resultados)

# ==========================
# 9️⃣ Gráficos Real vs Predicho
# ==========================

# Lineal
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_lineal, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Consumo Real")
plt.ylabel("Consumo Predicho")
plt.title("🔍 Regresión Lineal: Real vs Predicho")
plt.tight_layout()
plt.show()

# Polinómicos grado 2 y 3
for grado, (y_real, y_pred) in y_pred_poly_dict.items():
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_real, y=y_pred, alpha=0.5)
    plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], '--r')
    plt.xlabel("Consumo Real")
    plt.ylabel("Consumo Predicho")
    plt.title(f"📊 Polinómico grado {grado}: Real vs Predicho")
    plt.tight_layout()
    plt.show()

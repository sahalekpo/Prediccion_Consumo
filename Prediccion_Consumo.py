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
import os

# Crear directorio para guardar las gráficas si no existe
if not os.path.exists('graficas'):
    os.makedirs('graficas')

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

# Imputar valores nulos con la media (mantenido como solicitado)
df.fillna(df.mean(), inplace=True)

# Guardar copia original para comparación
df_original = df.copy()

# Visualizar boxplots antes de tratamiento de outliers
plt.figure(figsize=(12, 4))
for i, col in enumerate(df.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"{col} (antes)")
plt.tight_layout()
plt.savefig('graficas/1_boxplots_antes.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# ⭐ NUEVO MÉTODO: Transformación logarítmica para outliers
# ==========================
# Identificar outliers usando z-score
z_scores = np.abs(stats.zscore(df))
outliers_mask = z_scores >= 3

# Crear columnas transformadas solo para los outliers
df_transformed = df.copy()

# Aplicar transformación logarítmica solo a los valores positivos que son outliers
for col in df.columns:
    # Solo aplicamos la transformación a valores positivos (consumo y otras variables siempre deberían ser positivas)
    positive_outliers = (outliers_mask[col]) & (df[col] > 0)
    
    if positive_outliers.any():
        # Guardamos los valores originales para referencia
        original_values = df.loc[positive_outliers, col].copy()
        
        # Aplicamos transformación logarítmica a los outliers
        df_transformed.loc[positive_outliers, col] = np.log1p(df.loc[positive_outliers, col])
        
        print(f"\n🔄 Transformación logarítmica aplicada a {positive_outliers.sum()} outliers en '{col}'")
        print(f"   Ejemplo - Original: {original_values.iloc[0]:.2f}, Transformado: {df_transformed.loc[positive_outliers, col].iloc[0]:.2f}")

# Visualizar boxplots después de la transformación logarítmica
plt.figure(figsize=(12, 4))
for i, col in enumerate(df_transformed.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df_transformed[col])
    plt.title(f"{col} (después)")
plt.tight_layout()
plt.savefig('graficas/2_boxplots_despues.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogramas para comparar distribuciones antes y después
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    # Original
    plt.subplot(4, 2, i*2+1)
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} (Original)")
    
    # Transformado
    plt.subplot(4, 2, i*2+2)
    sns.histplot(df_transformed[col], kde=True)
    plt.title(f"{col} (Transformado)")
plt.tight_layout()
plt.savefig('graficas/2b_histogramas_comparacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Usaremos df_transformed para el modelado
df = df_transformed

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
plt.savefig('graficas/3_scatterplots.png', dpi=300, bbox_inches='tight')
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
plt.savefig('graficas/4_lineal_real_vs_predicho.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'graficas/5_polinomico_grado{grado}_real_vs_predicho.png', dpi=300, bbox_inches='tight')
    plt.show()

# Guardar también la tabla de resultados como imagen
plt.figure(figsize=(8, 3))
plt.axis('off')
tabla = plt.table(cellText=df_resultados.values, 
                 colLabels=df_resultados.columns, 
                 loc='center', 
                 cellLoc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.5)
plt.savefig('graficas/6_tabla_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 🔄 Análisis adicional: Comparación con datos originales
# ==========================
# Crear un gráfico para mostrar el efecto de la transformación en los outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(df_original.columns):
    plt.subplot(2, 2, i+1)
    plt.scatter(df_original[col], df_transformed[col], alpha=0.5)
    plt.plot([df_original[col].min(), df_original[col].max()], 
             [df_original[col].min(), df_original[col].max()], '--r')
    plt.xlabel(f"{col} Original")
    plt.ylabel(f"{col} Transformado")
    plt.title(f"Efecto de la transformación en {col}")
plt.tight_layout()
plt.savefig('graficas/7_comparacion_transformacion.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Todas las gráficas han sido guardadas en la carpeta 'graficas/'")
print("\n📝 Resumen del método aplicado:")
print("1. Se identificaron outliers usando z-scores (valores con z > 3)")
print("2. Se aplicó transformación logarítmica (log(1+x)) solo a los outliers")
print("3. Esta transformación reduce el impacto de valores extremos sin eliminarlos")
print("4. Los modelos se entrenaron con los datos transformados")
print("5. La transformación logarítmica es ideal para datos de consumo porque:")
print("   - Preserva la información de eventos reales (picos de consumo)")
print("   - Normaliza la distribución asimétrica típica de datos de consumo")
print("   - Mejora el rendimiento de los modelos lineales")
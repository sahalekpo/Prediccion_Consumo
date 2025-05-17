import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Configuración para visualizaciones
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 1. Carga y análisis inicial
print("1. CARGA Y ANÁLISIS INICIAL\n" + "="*30)
df = pd.read_csv('consumo_hogar.csv')

print("\nPrimeras filas del dataset:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 2. Limpieza de datos
print("\n\n2. LIMPIEZA DE DATOS\n" + "="*30)

# Guardar una copia del dataframe original para comparaciones
df_original = df.copy()

# Identificar valores nulos
print("\nPorcentaje de valores nulos por columna:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Imputación de valores nulos con la media
for col in df.columns:
    if df[col].isnull().sum() > 0:
        media = df[col].mean()
        print(f"Imputando valores nulos en '{col}' con la media: {media:.2f}")
        df[col].fillna(media, inplace=True)

# Verificar que no queden valores nulos
print("\nValores nulos después de la imputación:")
print(df.isnull().sum())

# Detección de outliers usando Z-score
print("\nDetección de outliers usando Z-score:")
z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
outliers_indices = np.where(abs_z_scores > 3)[0]
print(f"Número de outliers detectados (Z-score > 3): {len(outliers_indices)}")

# Identificar outliers en la columna de Consumo_kWh
z_consumo = np.abs(stats.zscore(df['Consumo_kWh']))
outliers_consumo = df[z_consumo > 3]
print(f"\nNúmero de outliers en Consumo_kWh: {len(outliers_consumo)}")
print("Ejemplos de outliers en Consumo_kWh:")
print(outliers_consumo.head())

# En lugar de eliminar los outliers, vamos a tratarlos con winsorization
def winsorize_column(df, column, limits=(0.01, 0.01)):
    """Aplica winsorization a una columna específica"""
    lower_limit = np.percentile(df[column], limits[0] * 100)
    upper_limit = np.percentile(df[column], 100 - limits[1] * 100)
    
    df_winsorized = df.copy()
    df_winsorized.loc[df_winsorized[column] < lower_limit, column] = lower_limit
    df_winsorized.loc[df_winsorized[column] > upper_limit, column] = upper_limit
    
    return df_winsorized

# Aplicar winsorization a la columna de Consumo_kWh
df_winsorized = winsorize_column(df, 'Consumo_kWh', limits=(0.01, 0.01))
print("\nEstadísticas de Consumo_kWh antes de winsorization:")
print(df['Consumo_kWh'].describe())
print("\nEstadísticas de Consumo_kWh después de winsorization:")
print(df_winsorized['Consumo_kWh'].describe())

# Usaremos el dataframe con winsorization para el resto del análisis
df = df_winsorized

# 3. Visualización exploratoria
print("\n\n3. VISUALIZACIÓN EXPLORATORIA\n" + "="*30)

# Crear una figura para los scatterplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatterplot: Temperatura vs Consumo
sns.scatterplot(x='Temperatura', y='Consumo_kWh', data=df, ax=axes[0])
axes[0].set_title('Temperatura vs Consumo Eléctrico')
axes[0].set_xlabel('Temperatura (°C)')
axes[0].set_ylabel('Consumo (kWh)')

# Scatterplot: Personas vs Consumo
sns.scatterplot(x='Personas', y='Consumo_kWh', data=df, ax=axes[1])
axes[1].set_title('Personas en el Hogar vs Consumo Eléctrico')
axes[1].set_xlabel('Número de Personas')
axes[1].set_ylabel('Consumo (kWh)')

# Scatterplot: Electrodomesticos vs Consumo
sns.scatterplot(x='Electrodomesticos', y='Consumo_kWh', data=df, ax=axes[2])
axes[2].set_title('Electrodomésticos en Uso vs Consumo Eléctrico')
axes[2].set_xlabel('Cantidad de Electrodomésticos')
axes[2].set_ylabel('Consumo (kWh)')

plt.tight_layout()
plt.show()  # Mostrar directamente en lugar de guardar

# Matriz de correlación
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()  # Mostrar directamente

print("Matriz de correlación:")
print(correlation_matrix)

# Boxplots para visualizar la distribución y outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(y=df['Temperatura'], ax=axes[0, 0])
axes[0, 0].set_title('Distribución de Temperatura')

sns.boxplot(y=df['Personas'], ax=axes[0, 1])
axes[0, 1].set_title('Distribución de Personas')

sns.boxplot(y=df['Electrodomesticos'], ax=axes[1, 0])
axes[1, 0].set_title('Distribución de Electrodomésticos')

sns.boxplot(y=df['Consumo_kWh'], ax=axes[1, 1])
axes[1, 1].set_title('Distribución de Consumo Eléctrico')

plt.tight_layout()
plt.show()  # Mostrar directamente

# 4. Modelado
print("\n\n4. MODELADO\n" + "="*30)

# Preparar los datos
X = df[['Temperatura', 'Personas', 'Electrodomesticos']]
y = df['Consumo_kWh']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a) Regresión lineal múltiple
print("\na) Regresión Lineal Múltiple")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Mostrar coeficientes e intercepto
print(f"Intercepto: {linear_model.intercept_:.4f}")
print("Coeficientes:")
for feature, coef in zip(X.columns, linear_model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Predicciones con el modelo lineal
y_pred_linear = linear_model.predict(X_test)

# b) Regresión polinómica de grado 2
print("\nb) Regresión Polinómica de Grado 2")
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2_train = poly_features_2.fit_transform(X_train)
X_poly_2_test = poly_features_2.transform(X_test)

poly_model_2 = LinearRegression()
poly_model_2.fit(X_poly_2_train, y_train)

# Predicciones con el modelo polinómico de grado 2
y_pred_poly_2 = poly_model_2.predict(X_poly_2_test)

# c) Regresión polinómica de grado 3
print("\nc) Regresión Polinómica de Grado 3")
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_3_train = poly_features_3.fit_transform(X_train)
X_poly_3_test = poly_features_3.transform(X_test)

poly_model_3 = LinearRegression()
poly_model_3.fit(X_poly_3_train, y_train)

# Predicciones con el modelo polinómico de grado 3
y_pred_poly_3 = poly_model_3.predict(X_poly_3_test)

# 5. Evaluación y comparación
print("\n\n5. EVALUACIÓN Y COMPARACIÓN\n" + "="*30)

# Calcular métricas para el modelo lineal
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Calcular métricas para el modelo polinómico de grado 2
mse_poly_2 = mean_squared_error(y_test, y_pred_poly_2)
r2_poly_2 = r2_score(y_test, y_pred_poly_2)

# Calcular métricas para el modelo polinómico de grado 3
mse_poly_3 = mean_squared_error(y_test, y_pred_poly_3)
r2_poly_3 = r2_score(y_test, y_pred_poly_3)

# Mostrar tabla de métricas
print("\nComparación de Modelos:")
print(f"{'Modelo':<25} {'MSE':>10} {'R²':>10}")
print(f"{'-'*25} {'-'*10} {'-'*10}")
print(f"{'Regresión Lineal':<25} {mse_linear:>10.4f} {r2_linear:>10.4f}")
print(f"{'Regresión Polinómica (2)':<25} {mse_poly_2:>10.4f} {r2_poly_2:>10.4f}")
print(f"{'Regresión Polinómica (3)':<25} {mse_poly_3:>10.4f} {r2_poly_3:>10.4f}")

# Graficar valores predichos vs reales
plt.figure(figsize=(18, 6))

# Modelo Lineal
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Regresión Lineal\nValores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.text(0.05, 0.95, f'MSE: {mse_linear:.4f}\nR²: {r2_linear:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

# Modelo Polinómico de Grado 2
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_poly_2, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Regresión Polinómica (Grado 2)\nValores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.text(0.05, 0.95, f'MSE: {mse_poly_2:.4f}\nR²: {r2_poly_2:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

# Modelo Polinómico de Grado 3
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_poly_3, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Regresión Polinómica (Grado 3)\nValores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.text(0.05, 0.95, f'MSE: {mse_poly_3:.4f}\nR²: {r2_poly_3:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()  # Mostrar directamente

# 6. Conclusiones
print("\n\n6. CONCLUSIONES\n" + "="*30)

# Determinar el mejor modelo basado en R²
models = {
    'Regresión Lineal': r2_linear,
    'Regresión Polinómica (Grado 2)': r2_poly_2,
    'Regresión Polinómica (Grado 3)': r2_poly_3
}
best_model = max(models, key=models.get)

print(f"\nEl mejor modelo según R² es: {best_model} con R² = {models[best_model]:.4f}")

# Analizar la importancia de las variables en el modelo lineal
print("\nImportancia de las variables en el modelo lineal:")
for feature, coef in zip(X.columns, linear_model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Verificar si hay sobreajuste en los modelos polinómicos
print("\nAnálisis de posible sobreajuste:")
# Calcular métricas en el conjunto de entrenamiento
y_train_pred_linear = linear_model.predict(X_train)
r2_train_linear = r2_score(y_train, y_train_pred_linear)

y_train_pred_poly_2 = poly_model_2.predict(X_poly_2_train)
r2_train_poly_2 = r2_score(y_train, y_train_pred_poly_2)

y_train_pred_poly_3 = poly_model_3.predict(X_poly_3_train)
r2_train_poly_3 = r2_score(y_train, y_train_pred_poly_3)

print(f"Diferencia R² (train-test) para Regresión Lineal: {r2_train_linear - r2_linear:.4f}")
print(f"Diferencia R² (train-test) para Regresión Polinómica (2): {r2_train_poly_2 - r2_poly_2:.4f}")
print(f"Diferencia R² (train-test) para Regresión Polinómica (3): {r2_train_poly_3 - r2_poly_3:.4f}")

# Recomendaciones para una empresa energética
print("\nRecomendaciones para una empresa energética:")
print("1. Basado en los coeficientes del modelo, se puede identificar qué factores tienen mayor")
print("   impacto en el consumo eléctrico y enfocar campañas de eficiencia energética en ellos.")
print("2. Utilizar el modelo predictivo para estimar la demanda futura y planificar la capacidad.")
print("3. Desarrollar tarifas personalizadas basadas en los patrones de consumo identificados.")
print("4. Implementar programas de incentivos para reducir el consumo en horas pico.")
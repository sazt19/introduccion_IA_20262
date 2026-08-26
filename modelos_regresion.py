# ============================================================
# COMPARACIÓN DE MODELOS DE REGRESIÓN
# ============================================================
# Modelos:
# 1. Regresión Lineal
# 2. Regresión Polinomial
# 3. Regresión Logarítmica
# 4. Decision Tree Regressor
#
# Dataset: California Housing
# Partición: 80% entrenamiento / 20% prueba
# Cross Validation: 5-Fold
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. CARGAR DATASET
# ============================================================

california = fetch_california_housing()

df = pd.DataFrame(
    california.data,
    columns=california.feature_names
)

df["MedHouseVal"] = california.target


print("=" * 75)
print("DATASET CALIFORNIA HOUSING")
print("=" * 75)

print("\nPrimeras filas:")
print(df.head())

print("\nDimensiones:")
print(df.shape)

print("\nVariables:")
print(list(df.columns))


# ============================================================
# 3. SELECCIÓN DE VARIABLES
# ============================================================

# Variable independiente
X = df[["MedInc"]]

# Variable dependiente
y = df["MedHouseVal"]


# ============================================================
# 4. PARTICIÓN 80/20
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 75)
print("PARTICIÓN DE LOS DATOS")
print("=" * 75)

print(f"\nTotal de datos: {len(X)}")
print(f"Entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Prueba: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")


# ============================================================
# 5. DEFINIR LOS MODELOS
# ============================================================

# ------------------------------------------------------------
# MODELO 1: REGRESIÓN LINEAL
# ------------------------------------------------------------

modelo_lineal = LinearRegression()


# ------------------------------------------------------------
# MODELO 2: REGRESIÓN POLINOMIAL
# ------------------------------------------------------------

modelo_polinomial = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(degree=2)
    ),
    (
        "linear_regression",
        LinearRegression()
    )
])


# ------------------------------------------------------------
# MODELO 3: REGRESIÓN LOGARÍTMICA
# ------------------------------------------------------------

# La regresión logarítmica se implementará
# transformando X mediante np.log.


class RegresionLogaritmica(BaseEstimator, RegressorMixin):

    def fit(self, X, y):

        self.modelo = LinearRegression()

        X_log = np.log(
            np.asarray(X)
        )

        self.modelo.fit(
            X_log,
            y
        )

        return self

    def predict(self, X):

        X_log = np.log(
            np.asarray(X)
        )

        return self.modelo.predict(
            X_log
        )


modelo_logaritmico = RegresionLogaritmica()


# ------------------------------------------------------------
# MODELO 4: DECISION TREE
# ------------------------------------------------------------

modelo_arbol = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# ============================================================
# 6. DICCIONARIO DE MODELOS
# ============================================================

modelos = {

    "Regresión Lineal":
        modelo_lineal,

    "Regresión Polinomial":
        modelo_polinomial,

    "Regresión Logarítmica":
        modelo_logaritmico,

    "Decision Tree":
        modelo_arbol
}


# ============================================================
# 7. ENTRENAMIENTO Y PREDICCIONES
# ============================================================

predicciones = {}

resultados = []


for nombre, modelo in modelos.items():

    print("\n" + "=" * 75)
    print(nombre.upper())
    print("=" * 75)

    # Entrenamiento
    modelo.fit(
        X_train,
        y_train
    )

    # Predicción
    y_pred = modelo.predict(
        X_test
    )

    predicciones[nombre] = y_pred

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"\nMAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    resultados.append({
        "Modelo": nombre,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    })


# ============================================================
# 8. TABLA DE MÉTRICAS
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)


print("\n\n" + "=" * 75)
print("TABLA DE MÉTRICAS")
print("=" * 75)

print(
    df_resultados.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. CROSS VALIDATION
# ============================================================

print("\n\n" + "=" * 75)
print("CROSS VALIDATION - 5 FOLDS")
print("=" * 75)


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


resultados_cv = []


for nombre, modelo in modelos.items():

    # cross_val_score utiliza R²
    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=kf,
        scoring="r2"
    )

    promedio = scores.mean()
    desviacion = scores.std()

    resultados_cv.append({
        "Modelo": nombre,
        "R² CV Promedio": promedio,
        "Desviación CV": desviacion
    })

    print(f"\n{nombre}")

    print(
        "Scores:",
        [round(score, 4) for score in scores]
    )

    print(
        f"R² promedio: {promedio:.4f}"
    )

    print(
        f"Desviación estándar: {desviacion:.4f}"
    )


# ============================================================
# 10. TABLA CROSS VALIDATION
# ============================================================

df_cv = pd.DataFrame(
    resultados_cv
)


print("\n\n" + "=" * 75)
print("RESULTADOS CROSS VALIDATION")
print("=" * 75)

print(
    df_cv.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 11. COMBINAR MÉTRICAS + CROSS VALIDATION
# ============================================================

df_final = df_resultados.merge(
    df_cv,
    on="Modelo"
)


print("\n\n" + "=" * 75)
print("COMPARACIÓN FINAL")
print("=" * 75)

print(
    df_final.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. PORCENTAJE DE DESEMPEÑO
# ============================================================

print("\n\n" + "=" * 75)
print("PORCENTAJE DE DESEMPEÑO")
print("=" * 75)

for _, fila in df_final.iterrows():

    porcentaje_r2 = fila["R²"] * 100

    porcentaje_cv = (
        fila["R² CV Promedio"] * 100
    )

    print(
        f"\n{fila['Modelo']}:"
    )

    print(
        f"  R²: {porcentaje_r2:.2f}%"
    )

    print(
        f"  R² Cross Validation: "
        f"{porcentaje_cv:.2f}%"
    )


# ============================================================
# 13. DETERMINAR EL MEJOR MODELO
# ============================================================

mejor_modelo = df_final.loc[
    df_final["R² CV Promedio"].idxmax()
]


print("\n\n" + "=" * 75)
print("🏆 MODELO GANADOR")
print("=" * 75)

print(
    f"\nModelo: {mejor_modelo['Modelo']}"
)

print(
    f"R² en prueba: "
    f"{mejor_modelo['R²']:.4f}"
)

print(
    f"Desempeño R²: "
    f"{mejor_modelo['R²'] * 100:.2f}%"
)

print(
    f"R² Cross Validation: "
    f"{mejor_modelo['R² CV Promedio'] * 100:.2f}%"
)

print(
    f"RMSE: "
    f"{mejor_modelo['RMSE']:.4f}"
)

print(
    f"MAE: "
    f"{mejor_modelo['MAE']:.4f}"
)


# ============================================================
# 14. CREAR DATOS PARA LAS LÍNEAS DE ESTIMACIÓN
# ============================================================

X_grafica = np.linspace(
    X["MedInc"].min(),
    X["MedInc"].max(),
    500
).reshape(-1, 1)


# ============================================================
# 15. LÍNEA DE ESTIMACIÓN - REGRESIÓN LINEAL
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["MedInc"],
    y_test,
    alpha=0.3,
    label="Datos reales"
)

plt.plot(
    X_grafica,
    modelo_lineal.predict(X_grafica),
    linewidth=3,
    label="Línea de estimación"
)

plt.xlabel(
    "Ingreso medio (MedInc)"
)

plt.ylabel(
    "Valor medio de vivienda (MedHouseVal)"
)

plt.title(
    "Regresión Lineal - Línea de Estimación"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 16. LÍNEA DE ESTIMACIÓN - REGRESIÓN POLINOMIAL
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["MedInc"],
    y_test,
    alpha=0.3,
    label="Datos reales"
)

plt.plot(
    X_grafica,
    modelo_polinomial.predict(X_grafica),
    linewidth=3,
    label="Curva de estimación"
)

plt.xlabel(
    "Ingreso medio (MedInc)"
)

plt.ylabel(
    "Valor medio de vivienda (MedHouseVal)"
)

plt.title(
    "Regresión Polinomial - Curva de Estimación"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 17. LÍNEA DE ESTIMACIÓN - REGRESIÓN LOGARÍTMICA
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["MedInc"],
    y_test,
    alpha=0.3,
    label="Datos reales"
)

plt.plot(
    X_grafica,
    modelo_logaritmico.predict(X_grafica),
    linewidth=3,
    label="Curva de estimación"
)

plt.xlabel(
    "Ingreso medio (MedInc)"
)

plt.ylabel(
    "Valor medio de vivienda (MedHouseVal)"
)

plt.title(
    "Regresión Logarítmica - Curva de Estimación"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 18. LÍNEA DE ESTIMACIÓN - DECISION TREE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["MedInc"],
    y_test,
    alpha=0.3,
    label="Datos reales"
)

plt.plot(
    X_grafica,
    modelo_arbol.predict(X_grafica),
    linewidth=3,
    label="Estimación Decision Tree"
)

plt.xlabel(
    "Ingreso medio (MedInc)"
)

plt.ylabel(
    "Valor medio de vivienda (MedHouseVal)"
)

plt.title(
    "Decision Tree Regressor - Línea de Estimación"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 19. COMPARACIÓN DE R²
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df_final["Modelo"],
    df_final["R²"]
)

plt.ylabel(
    "R²"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Comparación de R²"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 20. COMPARACIÓN DE CROSS VALIDATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df_final["Modelo"],
    df_final["R² CV Promedio"]
)

plt.ylabel(
    "R² promedio"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Cross Validation - R² promedio (5 folds)"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 21. COMPARACIÓN FINAL EN PORCENTAJES
# ============================================================

plt.figure(figsize=(10, 6))

porcentajes = (
    df_final["R² CV Promedio"] * 100
)

plt.bar(
    df_final["Modelo"],
    porcentajes
)

plt.ylabel(
    "Desempeño (%)"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Desempeño de los modelos según Cross Validation"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# FIN DEL PROGRAMA
# ============================================================

print("\n" + "=" * 75)
print("PROCESO FINALIZADO")
print("=" * 75)
# ============================================================
# CLASIFICACIÓN DEL DATASET IRIS
# ANÁLISIS DE SUBAJUSTE Y SOBREAJUSTE
# ============================================================
#
# Modelos:
# 1. LDA
# 2. K-NN (k=3)
# 3. K-NN (k=7)
# 4. Decision Tree normal
# 5. Decision Tree complejo
#
# Opciones:
# - Partición 50/50
# - Partición 40/60
# - Cross Validation
#
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. CARGAR DATASET IRIS
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names


print("=" * 75)
print("DATASET IRIS")
print("=" * 75)

print("\nNúmero de muestras:", len(X))

print("Número de características:", X.shape[1])

print("Características:")
for feature in feature_names:
    print("-", feature)

print("\nClases:")
for clase in class_names:
    print("-", clase)


# ============================================================
# 3. SELECCIONAR DOS VARIABLES PARA LAS FRONTERAS
# ============================================================

# Para poder visualizar las fronteras de decisión
# utilizaremos solamente:
#
# X[:, 2] = Petal Length
# X[:, 3] = Petal Width

X_visual = X[:, 2:4]

visual_feature_names = feature_names[2:4]


# ============================================================
# 4. NORMALIZACIÓN
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_visual)


# ============================================================
# 5. DEFINIR MODELOS
# ============================================================

modelos = {

    # Modelo original
    "LDA": LinearDiscriminantAnalysis(),

    # K-NN original
    "K-NN (k=3)": KNeighborsClassifier(
        n_neighbors=3
    ),

    # K-NN con mayor K
    "K-NN (k=7)": KNeighborsClassifier(
        n_neighbors=7
    ),

    # Árbol controlado
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    # Árbol deliberadamente complejo
    "Decision Tree complejo": DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )
}


# ============================================================
# 6. FUNCIÓN PARA EVALUAR UN MODELO
# ============================================================

def evaluar_modelo(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):

    # Entrenar
    modelo.fit(
        X_train,
        y_train
    )

    # Predicciones
    y_train_pred = modelo.predict(
        X_train
    )

    y_test_pred = modelo.predict(
        X_test
    )

    # Métricas entrenamiento
    accuracy_train = accuracy_score(
        y_train,
        y_train_pred
    )

    precision_train = precision_score(
        y_train,
        y_train_pred,
        average="weighted"
    )

    recall_train = recall_score(
        y_train,
        y_train_pred,
        average="weighted"
    )

    f1_train = f1_score(
        y_train,
        y_train_pred,
        average="weighted"
    )

    # Métricas prueba
    accuracy_test = accuracy_score(
        y_test,
        y_test_pred
    )

    precision_test = precision_score(
        y_test,
        y_test_pred,
        average="weighted"
    )

    recall_test = recall_score(
        y_test,
        y_test_pred,
        average="weighted"
    )

    f1_test = f1_score(
        y_test,
        y_test_pred,
        average="weighted"
    )

    return {

        "Modelo": "",
        
        "Accuracy Train": accuracy_train,
        "Accuracy Test": accuracy_test,

        "Precision Train": precision_train,
        "Precision Test": precision_test,

        "Recall Train": recall_train,
        "Recall Test": recall_test,

        "F1 Train": f1_train,
        "F1 Test": f1_test,

        "Predicciones": y_test_pred
    }


# ============================================================
# 7. FUNCIÓN PARA CROSS VALIDATION
# ============================================================

def realizar_cross_validation(
    modelo,
    X,
    y
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    return scores


# ============================================================
# 8. FUNCIÓN PARA MOSTRAR RESULTADOS
# ============================================================

def ejecutar_experimento(
    nombre_particion,
    test_size
):

    print("\n\n")
    print("=" * 75)
    print(f"PARTICIÓN {nombre_particion}")
    print("=" * 75)

    # --------------------------------------------------------
    # PARTICIÓN
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled,
        y,

        test_size=test_size,

        random_state=42,

        stratify=y
    )

    print(
        f"\nEntrenamiento: {len(X_train)} datos"
    )

    print(
        f"Prueba: {len(X_test)} datos"
    )

    resultados = []

    modelos_entrenados = {}

    # --------------------------------------------------------
    # ENTRENAR MODELOS
    # --------------------------------------------------------

    for nombre, modelo in modelos.items():

        resultado = evaluar_modelo(

            modelo,
            X_train,
            X_test,
            y_train,
            y_test
        )

        resultado["Modelo"] = nombre

        resultados.append(resultado)

        modelos_entrenados[nombre] = modelo

        # ----------------------------------------------------
        # MOSTRAR RESULTADOS
        # ----------------------------------------------------

        print("\n" + "-" * 75)

        print(nombre)

        print("-" * 75)

        print(
            f"Accuracy entrenamiento: "
            f"{resultado['Accuracy Train'] * 100:.2f}%"
        )

        print(
            f"Accuracy prueba: "
            f"{resultado['Accuracy Test'] * 100:.2f}%"
        )

        print(
            f"Precision prueba: "
            f"{resultado['Precision Test'] * 100:.2f}%"
        )

        print(
            f"Recall prueba: "
            f"{resultado['Recall Test'] * 100:.2f}%"
        )

        print(
            f"F1 prueba: "
            f"{resultado['F1 Test'] * 100:.2f}%"
        )

        # Diferencia entrenamiento/prueba
        diferencia = (
            resultado["Accuracy Train"]
            -
            resultado["Accuracy Test"]
        )

        print(
            f"Diferencia Train-Test: "
            f"{diferencia * 100:.2f}%"
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    print("\n\n")
    print("=" * 75)
    print(
        f"TABLA DE RESULTADOS - {nombre_particion}"
    )
    print("=" * 75)

    columnas = [
        "Modelo",
        "Accuracy Train",
        "Accuracy Test",
        "Precision Test",
        "Recall Test",
        "F1 Test"
    ]

    print(
        df_resultados[columnas].to_string(
            index=False,
            float_format=lambda x: f"{x * 100:.2f}%"
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        df_resultados,
        modelos_entrenados
    )


# ============================================================
# 9. EXPERIMENTO 1: 50% / 50%
# ============================================================

(
    X_train_50,
    X_test_50,
    y_train_50,
    y_test_50,
    resultados_50,
    modelos_50
) = ejecutar_experimento(
    "50% entrenamiento / 50% prueba",
    0.50
)


# ============================================================
# 10. EXPERIMENTO 2: 40% / 60%
# ============================================================

(
    X_train_40,
    X_test_40,
    y_train_40,
    y_test_40,
    resultados_40,
    modelos_40
) = ejecutar_experimento(
    "40% entrenamiento / 60% prueba",
    0.60
)


# ============================================================
# 11. CROSS VALIDATION
# ============================================================

print("\n\n")
print("=" * 75)
print("CROSS VALIDATION - 5 FOLDS")
print("=" * 75)

resultados_cv = []

for nombre, modelo in modelos.items():

    scores = realizar_cross_validation(
        modelo,
        X_scaled,
        y
    )

    promedio = scores.mean()

    desviacion = scores.std()

    resultados_cv.append({

        "Modelo": nombre,

        "CV Promedio": promedio,

        "CV Desviacion": desviacion
    })

    print("\n" + "-" * 75)

    print(nombre)

    print(
        "Scores:",
        [
            f"{score * 100:.2f}%"
            for score in scores
        ]
    )

    print(
        f"Promedio: "
        f"{promedio * 100:.2f}%"
    )

    print(
        f"Desviación: "
        f"{desviacion * 100:.2f}%"
    )


df_cv = pd.DataFrame(
    resultados_cv
)


# ============================================================
# 12. COMPARACIÓN DE ACCURACY
# ============================================================

plt.figure(figsize=(12, 6))

x = np.arange(
    len(modelos)
)

ancho = 0.25

plt.bar(
    x - ancho,
    resultados_50["Accuracy Train"],
    ancho,
    label="Train 50/50"
)

plt.bar(
    x,
    resultados_50["Accuracy Test"],
    ancho,
    label="Test 50/50"
)

plt.bar(
    x + ancho,
    resultados_40["Accuracy Test"],
    ancho,
    label="Test 40/60"
)

plt.xticks(
    x,
    resultados_50["Modelo"],
    rotation=20
)

plt.ylabel("Accuracy")

plt.title(
    "Comparación de Accuracy - Entrenamiento y Prueba"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. GRÁFICA CROSS VALIDATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df_cv["Modelo"],
    df_cv["CV Promedio"]
)

plt.ylabel(
    "Accuracy promedio"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Cross Validation - 5 Folds"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. FRONTERAS DE DECISIÓN
# ============================================================

def graficar_frontera(
    modelo,
    X,
    y,
    titulo
):

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5

    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(

        np.arange(
            x_min,
            x_max,
            0.02
        ),

        np.arange(
            y_min,
            y_max,
            0.02
        )
    )

    Z = modelo.predict(

        np.c_[
            xx.ravel(),
            yy.ravel()
        ]
    )

    Z = Z.reshape(
        xx.shape
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="black"
    )

    plt.xlabel(
        visual_feature_names[0]
    )

    plt.ylabel(
        visual_feature_names[1]
    )

    plt.title(
        titulo
    )

    plt.grid(
        True,
        alpha=0.2
    )

    plt.show()


# ============================================================
# 15. FRONTERAS PARA LOS MODELOS
# ============================================================

for nombre, modelo in modelos_50.items():

    graficar_frontera(

        modelo,

        X_train_50,

        y_train_50,

        f"Frontera de decisión - {nombre} - 50/50"
    )


# ============================================================
# 16. MATRICES DE CONFUSIÓN
# ============================================================

for nombre, modelo in modelos_50.items():

    y_pred = modelo.predict(
        X_test_50
    )

    cm = confusion_matrix(
        y_test_50,
        y_pred
    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=class_names
    )

    display.plot()

    plt.title(
        f"Matriz de confusión - {nombre}"
    )

    plt.show()


# ============================================================
# 17. ANÁLISIS DE SOBREAJUSTE
# ============================================================

print("\n\n")
print("=" * 75)
print("ANÁLISIS DE SOBREAJUSTE")
print("=" * 75)

for _, fila in resultados_50.iterrows():

    diferencia = (
        fila["Accuracy Train"]
        -
        fila["Accuracy Test"]
    )

    print(
        f"\n{fila['Modelo']}"
    )

    print(
        f"Train: "
        f"{fila['Accuracy Train'] * 100:.2f}%"
    )

    print(
        f"Test: "
        f"{fila['Accuracy Test'] * 100:.2f}%"
    )

    print(
        f"Diferencia: "
        f"{diferencia * 100:.2f}%"
    )

    if diferencia > 0.10:

        print(
            "⚠ Posible SOBREAJUSTE"
        )

    elif (
        fila["Accuracy Train"] < 0.80
        and
        fila["Accuracy Test"] < 0.80
    ):

        print(
            "⚠ Posible SUBAJUSTE"
        )

    else:

        print(
            "✓ Ajuste razonable"
        )


# ============================================================
# 18. COMPARAR K-NN 3 VS K-NN 7
# ============================================================

print("\n\n")
print("=" * 75)
print("COMPARACIÓN K-NN: k=3 VS k=7")
print("=" * 75)

knn3 = resultados_50[
    resultados_50["Modelo"] == "K-NN (k=3)"
].iloc[0]

knn7 = resultados_50[
    resultados_50["Modelo"] == "K-NN (k=7)"
].iloc[0]


print("\nK-NN k=3")

print(
    f"Train: "
    f"{knn3['Accuracy Train'] * 100:.2f}%"
)

print(
    f"Test: "
    f"{knn3['Accuracy Test'] * 100:.2f}%"
)


print("\nK-NN k=7")

print(
    f"Train: "
    f"{knn7['Accuracy Train'] * 100:.2f}%"
)

print(
    f"Test: "
    f"{knn7['Accuracy Test'] * 100:.2f}%"
)


# ============================================================
# 19. MEJOR MODELO SEGÚN CROSS VALIDATION
# ============================================================

mejor_cv = df_cv.loc[
    df_cv["CV Promedio"].idxmax()
]


print("\n\n")
print("=" * 75)
print("🏆 MEJOR MODELO SEGÚN CROSS VALIDATION")
print("=" * 75)

print(
    f"\nModelo: "
    f"{mejor_cv['Modelo']}"
)

print(
    f"Accuracy promedio: "
    f"{mejor_cv['CV Promedio'] * 100:.2f}%"
)

print(
    f"Desviación estándar: "
    f"{mejor_cv['CV Desviacion'] * 100:.2f}%"
)


# ============================================================
# 20. RESUMEN FINAL
# ============================================================

print("\n\n")
print("=" * 75)
print("RESUMEN FINAL")
print("=" * 75)

print("""
SUBAJUSTE:
Ocurre cuando el modelo es demasiado simple y
obtiene malos resultados tanto en entrenamiento
como en prueba.

SOBREAJUSTE:
Ocurre cuando el modelo aprende demasiado los datos
de entrenamiento y su rendimiento disminuye
considerablemente en los datos de prueba.

K-NN:
Aumentar k de 3 a 7 hace que el modelo considere
más vecinos para realizar la clasificación.

DECISION TREE:
Un árbol sin límite de profundidad puede volverse
muy complejo y memorizar los datos de entrenamiento.

CROSS VALIDATION:
Permite evaluar el modelo en diferentes particiones
y obtener una estimación más estable de su rendimiento.
""")

print("\nProceso terminado.")
# ============================================================
# COMPARACIÓN DE CLASIFICADORES
# ============================================================
#
# DATASETS:
# 1. MNIST Digits
# 2. MNIST Fashion
# 3. Wine
# 4. Breast Cancer
#
# MODELOS:
# 1. LDA
# 2. K-NN k=3
# 3. K-NN k=7
# 4. Decision Tree
# 5. Decision Tree complejo
#
# VALIDACIONES:
# 1. 50/50
# 2. 40/60
# 3. 5-Fold Cross Validation
# 4. Leave-One-Out Cross Validation
#
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_digits,
    load_wine,
    load_breast_cancer,
    fetch_openml
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    LeaveOneOut
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

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
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42

# Cantidad de datos utilizados para LOOCV
# Se limita para que sea computacionalmente viable.
LOOCV_SAMPLES = 300


# ============================================================
# 3. CARGAR DATASETS
# ============================================================

print("=" * 80)
print("CARGANDO DATASETS")
print("=" * 80)


# ------------------------------------------------------------
# MNIST DIGITS
# ------------------------------------------------------------

digits = load_digits()

X_digits = digits.data
y_digits = digits.target

print(
    f"\nMNIST Digits: "
    f"{X_digits.shape[0]} muestras, "
    f"{X_digits.shape[1]} características"
)


# ------------------------------------------------------------
# MNIST FASHION
# ------------------------------------------------------------

print("\nDescargando/cargando Fashion-MNIST...")

fashion = fetch_openml(
    "Fashion-MNIST",
    version=1,
    as_frame=False
)

X_fashion = fashion.data.astype(np.float32)
y_fashion = fashion.target.astype(int)

print(
    f"MNIST Fashion: "
    f"{X_fashion.shape[0]} muestras, "
    f"{X_fashion.shape[1]} características"
)


# ------------------------------------------------------------
# WINE
# ------------------------------------------------------------

wine = load_wine()

X_wine = wine.data
y_wine = wine.target

print(
    f"\nWine: "
    f"{X_wine.shape[0]} muestras, "
    f"{X_wine.shape[1]} características"
)


# ------------------------------------------------------------
# BREAST CANCER
# ------------------------------------------------------------

cancer = load_breast_cancer()

X_cancer = cancer.data
y_cancer = cancer.target

print(
    f"Breast Cancer: "
    f"{X_cancer.shape[0]} muestras, "
    f"{X_cancer.shape[1]} características"
)


# ============================================================
# 4. DICCIONARIO DE DATASETS
# ============================================================

datasets = {

    "MNIST Digits": (
        X_digits,
        y_digits
    ),

    "MNIST Fashion": (
        X_fashion,
        y_fashion
    ),

    "Wine": (
        X_wine,
        y_wine
    ),

    "Breast Cancer": (
        X_cancer,
        y_cancer
    )
}


# ============================================================
# 5. DEFINIR LOS CLASIFICADORES
# ============================================================

def crear_modelos():

    modelos = {

        # ----------------------------------------------------
        # LDA
        # ----------------------------------------------------

        "LDA": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "lda",
                LinearDiscriminantAnalysis()
            )
        ]),


        # ----------------------------------------------------
        # K-NN k=3
        # ----------------------------------------------------

        "K-NN (k=3)": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=3
                )
            )
        ]),


        # ----------------------------------------------------
        # K-NN k=7
        # ----------------------------------------------------

        "K-NN (k=7)": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=7
                )
            )
        ]),


        # ----------------------------------------------------
        # DECISION TREE NORMAL
        # ----------------------------------------------------

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ),


        # ----------------------------------------------------
        # DECISION TREE COMPLEJO
        # ----------------------------------------------------

        "Decision Tree complejo": DecisionTreeClassifier(
            max_depth=None,
            random_state=RANDOM_STATE
        )
    }

    return modelos


# ============================================================
# 6. FUNCIÓN PARA EVALUAR MODELO
# ============================================================

def evaluar_modelo(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):

    # Entrenamiento
    modelo.fit(
        X_train,
        y_train
    )

    # Predicciones entrenamiento
    y_train_pred = modelo.predict(
        X_train
    )

    # Predicciones prueba
    y_test_pred = modelo.predict(
        X_test
    )


    # --------------------------------------------------------
    # MÉTRICAS TRAIN
    # --------------------------------------------------------

    accuracy_train = accuracy_score(
        y_train,
        y_train_pred
    )


    precision_train = precision_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )


    recall_train = recall_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )


    f1_train = f1_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # MÉTRICAS TEST
    # --------------------------------------------------------

    accuracy_test = accuracy_score(
        y_test,
        y_test_pred
    )


    precision_test = precision_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )


    recall_test = recall_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )


    f1_test = f1_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )


    return {

        "Accuracy Train": accuracy_train,

        "Accuracy Test": accuracy_test,

        "Precision Test": precision_test,

        "Recall Test": recall_test,

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
        random_state=RANDOM_STATE
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
# 8. FUNCIÓN PARA LOOCV
# ============================================================

def realizar_loocv(
    modelo,
    X,
    y
):

    # --------------------------------------------------------
    # LIMITAR LA CANTIDAD DE DATOS
    # --------------------------------------------------------

    cantidad = min(
        LOOCV_SAMPLES,
        len(X)
    )


    # Seleccionar índices aleatorios
    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X),
        size=cantidad,
        replace=False
    )


    X_sample = X[indices]
    y_sample = y[indices]


    # --------------------------------------------------------
    # LOOCV
    # --------------------------------------------------------

    loo = LeaveOneOut()

    scores = cross_val_score(
        modelo,
        X_sample,
        y_sample,
        cv=loo,
        scoring="accuracy",
        n_jobs=-1
    )


    return scores


# ============================================================
# 9. EJECUTAR EXPERIMENTOS
# ============================================================

resultados_totales = []


for nombre_dataset, datos in datasets.items():

    X, y = datos


    print("\n\n")
    print("=" * 80)
    print(
        f"DATASET: {nombre_dataset}"
    )
    print("=" * 80)


    modelos = crear_modelos()


    # ========================================================
    # EXPERIMENTO 50/50
    # ========================================================

    print("\n")
    print("-" * 80)
    print("PARTICIÓN 50/50")
    print("-" * 80)


    X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(

        X,
        y,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=y
    )


    for nombre_modelo, modelo in modelos.items():

        resultado = evaluar_modelo(

            modelo,

            X_train_50,
            X_test_50,

            y_train_50,
            y_test_50
        )


        diferencia = (
            resultado["Accuracy Train"]
            -
            resultado["Accuracy Test"]
        )


        print(
            f"\n{nombre_modelo}"
        )

        print(
            f"Train: "
            f"{resultado['Accuracy Train'] * 100:.2f}%"
        )

        print(
            f"Test: "
            f"{resultado['Accuracy Test'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{resultado['Precision Test'] * 100:.2f}%"
        )

        print(
            f"Recall: "
            f"{resultado['Recall Test'] * 100:.2f}%"
        )

        print(
            f"F1: "
            f"{resultado['F1 Test'] * 100:.2f}%"
        )

        print(
            f"Diferencia Train-Test: "
            f"{diferencia * 100:.2f}%"
        )


        resultados_totales.append({

            "Dataset": nombre_dataset,

            "Modelo": nombre_modelo,

            "Partición": "50/50",

            "Accuracy Train":
                resultado["Accuracy Train"],

            "Accuracy Test":
                resultado["Accuracy Test"],

            "Precision":
                resultado["Precision Test"],

            "Recall":
                resultado["Recall Test"],

            "F1":
                resultado["F1 Test"]
        })


    # ========================================================
    # EXPERIMENTO 40/60
    # ========================================================

    print("\n")
    print("-" * 80)
    print("PARTICIÓN 40/60")
    print("-" * 80)


    X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(

        X,
        y,

        test_size=0.60,

        random_state=RANDOM_STATE,

        stratify=y
    )


    for nombre_modelo, modelo in modelos.items():

        resultado = evaluar_modelo(

            modelo,

            X_train_40,
            X_test_40,

            y_train_40,
            y_test_40
        )


        diferencia = (
            resultado["Accuracy Train"]
            -
            resultado["Accuracy Test"]
        )


        print(
            f"\n{nombre_modelo}"
        )

        print(
            f"Train: "
            f"{resultado['Accuracy Train'] * 100:.2f}%"
        )

        print(
            f"Test: "
            f"{resultado['Accuracy Test'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{resultado['Precision Test'] * 100:.2f}%"
        )

        print(
            f"Recall: "
            f"{resultado['Recall Test'] * 100:.2f}%"
        )

        print(
            f"F1: "
            f"{resultado['F1 Test'] * 100:.2f}%"
        )

        print(
            f"Diferencia Train-Test: "
            f"{diferencia * 100:.2f}%"
        )


        resultados_totales.append({

            "Dataset": nombre_dataset,

            "Modelo": nombre_modelo,

            "Partición": "40/60",

            "Accuracy Train":
                resultado["Accuracy Train"],

            "Accuracy Test":
                resultado["Accuracy Test"],

            "Precision":
                resultado["Precision Test"],

            "Recall":
                resultado["Recall Test"],

            "F1":
                resultado["F1 Test"]
        })


    # ========================================================
    # CROSS VALIDATION 5-FOLD
    # ========================================================

    print("\n")
    print("-" * 80)
    print("CROSS VALIDATION - 5 FOLD")
    print("-" * 80)


    for nombre_modelo, modelo in modelos.items():

        scores = realizar_cross_validation(

            modelo,
            X,
            y
        )


        promedio = scores.mean()

        desviacion = scores.std()


        print(
            f"\n{nombre_modelo}"
        )

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


        resultados_totales.append({

            "Dataset": nombre_dataset,

            "Modelo": nombre_modelo,

            "Partición": "5-Fold CV",

            "Accuracy Train": np.nan,

            "Accuracy Test": promedio,

            "Precision": np.nan,

            "Recall": np.nan,

            "F1": np.nan
        })


    # ========================================================
    # LEAVE-ONE-OUT
    # ========================================================

    print("\n")
    print("-" * 80)
    print(
        f"LEAVE-ONE-OUT "
        f"(máximo {LOOCV_SAMPLES} muestras)"
    )
    print("-" * 80)


    for nombre_modelo, modelo in modelos.items():

        scores = realizar_loocv(

            modelo,
            X,
            y
        )


        promedio = scores.mean()


        print(
            f"\n{nombre_modelo}"
        )

        print(
            f"LOOCV Accuracy: "
            f"{promedio * 100:.2f}%"
        )

        print(
            f"Muestras utilizadas: "
            f"{len(scores)}"
        )


        resultados_totales.append({

            "Dataset": nombre_dataset,

            "Modelo": nombre_modelo,

            "Partición": "LOOCV",

            "Accuracy Train": np.nan,

            "Accuracy Test": promedio,

            "Precision": np.nan,

            "Recall": np.nan,

            "F1": np.nan
        })


# ============================================================
# 10. CREAR TABLA GENERAL
# ============================================================

df_resultados = pd.DataFrame(
    resultados_totales
)


print("\n\n")
print("=" * 80)
print("TABLA GENERAL DE RESULTADOS")
print("=" * 80)


print(
    df_resultados.to_string(
        index=False,
        float_format=lambda x: f"{x * 100:.2f}%"
    )
)


# ============================================================
# 11. MEJORES MODELOS POR DATASET
# ============================================================

print("\n\n")
print("=" * 80)
print("MEJOR MODELO POR DATASET")
print("=" * 80)


# Solo utilizamos Cross Validation
# para decidir el modelo ganador.

df_cv = df_resultados[
    df_resultados["Partición"] == "5-Fold CV"
]


for dataset in datasets.keys():

    datos_dataset = df_cv[
        df_cv["Dataset"] == dataset
    ]


    mejor = datos_dataset.loc[
        datos_dataset["Accuracy Test"].idxmax()
    ]


    print(
        f"\n{dataset}"
    )

    print(
        f"🏆 Modelo: "
        f"{mejor['Modelo']}"
    )

    print(
        f"Accuracy CV: "
        f"{mejor['Accuracy Test'] * 100:.2f}%"
    )


# ============================================================
# 12. GRÁFICA COMPARATIVA 50/50
# ============================================================

for dataset in datasets.keys():

    datos = df_resultados[
        (df_resultados["Dataset"] == dataset)
        &
        (df_resultados["Partición"] == "50/50")
    ]


    plt.figure(
        figsize=(10, 6)
    )


    x = np.arange(
        len(datos)
    )


    ancho = 0.35


    plt.bar(
        x - ancho / 2,

        datos["Accuracy Train"],

        ancho,

        label="Entrenamiento"
    )


    plt.bar(
        x + ancho / 2,

        datos["Accuracy Test"],

        ancho,

        label="Prueba"
    )


    plt.xticks(
        x,
        datos["Modelo"],
        rotation=20
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.xlabel(
        "Modelo"
    )


    plt.title(
        f"{dataset} - Partición 50/50"
    )


    plt.legend()


    plt.grid(
        axis="y"
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# 13. GRÁFICA COMPARATIVA 40/60
# ============================================================

for dataset in datasets.keys():

    datos = df_resultados[
        (df_resultados["Dataset"] == dataset)
        &
        (df_resultados["Partición"] == "40/60")
    ]


    plt.figure(
        figsize=(10, 6)
    )


    x = np.arange(
        len(datos)
    )


    ancho = 0.35


    plt.bar(
        x - ancho / 2,

        datos["Accuracy Train"],

        ancho,

        label="Entrenamiento"
    )


    plt.bar(
        x + ancho / 2,

        datos["Accuracy Test"],

        ancho,

        label="Prueba"
    )


    plt.xticks(
        x,
        datos["Modelo"],
        rotation=20
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.xlabel(
        "Modelo"
    )


    plt.title(
        f"{dataset} - Partición 40/60"
    )


    plt.legend()


    plt.grid(
        axis="y"
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# 14. GRÁFICA CROSS VALIDATION
# ============================================================

for dataset in datasets.keys():

    datos = df_resultados[
        (df_resultados["Dataset"] == dataset)
        &
        (df_resultados["Partición"] == "5-Fold CV")
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        datos["Modelo"],

        datos["Accuracy Test"]
    )


    plt.ylabel(
        "Accuracy promedio"
    )


    plt.xlabel(
        "Modelo"
    )


    plt.title(
        f"{dataset} - 5-Fold Cross Validation"
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
# 15. GRÁFICA LOOCV
# ============================================================

for dataset in datasets.keys():

    datos = df_resultados[
        (df_resultados["Dataset"] == dataset)
        &
        (df_resultados["Partición"] == "LOOCV")
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        datos["Modelo"],

        datos["Accuracy Test"]
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.xlabel(
        "Modelo"
    )


    plt.title(
        f"{dataset} - Leave-One-Out Cross Validation"
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
# 16. ANÁLISIS DE SOBREAJUSTE
# ============================================================

print("\n\n")
print("=" * 80)
print("ANÁLISIS DE SUBAJUSTE Y SOBREAJUSTE")
print("=" * 80)


datos_5050 = df_resultados[
    df_resultados["Partición"] == "50/50"
]


for dataset in datasets.keys():

    print("\n")
    print("-" * 80)

    print(
        f"DATASET: {dataset}"
    )

    print("-" * 80)


    datos = datos_5050[
        datos_5050["Dataset"] == dataset
    ]


    for _, fila in datos.iterrows():

        train = fila["Accuracy Train"]

        test = fila["Accuracy Test"]

        diferencia = train - test


        print(
            f"\n{fila['Modelo']}"
        )

        print(
            f"Train: "
            f"{train * 100:.2f}%"
        )

        print(
            f"Test: "
            f"{test * 100:.2f}%"
        )

        print(
            f"Diferencia: "
            f"{diferencia * 100:.2f}%"
        )


        # ----------------------------------------------------
        # REGLAS DE INTERPRETACIÓN
        # ----------------------------------------------------

        if (
            train < 0.75
            and
            test < 0.75
        ):

            print(
                "⚠ Posible SUBAJUSTE"
            )


        elif diferencia > 0.10:

            print(
                "⚠ Posible SOBREAJUSTE"
            )


        else:

            print(
                "✓ Ajuste razonable"
            )


# ============================================================
# 17. COMPARACIÓN K-NN 3 VS K-NN 7
# ============================================================

print("\n\n")
print("=" * 80)
print("K-NN: COMPARACIÓN k=3 VS k=7")
print("=" * 80)


datos_cv = df_resultados[
    df_resultados["Partición"] == "5-Fold CV"
]


for dataset in datasets.keys():

    datos = datos_cv[
        datos_cv["Dataset"] == dataset
    ]


    knn3 = datos[
        datos["Modelo"] == "K-NN (k=3)"
    ]["Accuracy Test"].iloc[0]


    knn7 = datos[
        datos["Modelo"] == "K-NN (k=7)"
    ]["Accuracy Test"].iloc[0]


    print(
        f"\n{dataset}"
    )


    print(
        f"K-NN k=3: "
        f"{knn3 * 100:.2f}%"
    )


    print(
        f"K-NN k=7: "
        f"{knn7 * 100:.2f}%"
    )


    diferencia = (
        knn7 - knn3
    )


    if diferencia > 0:

        print(
            f"k=7 mejora "
            f"{diferencia * 100:.2f} puntos porcentuales"
        )

    elif diferencia < 0:

        print(
            f"k=7 disminuye "
            f"{abs(diferencia) * 100:.2f} puntos porcentuales"
        )

    else:

        print(
            "Ambos modelos tienen el mismo resultado"
        )


# ============================================================
# 18. MATRICES DE CONFUSIÓN
# ============================================================

print("\n\n")
print("=" * 80)
print("MATRICES DE CONFUSIÓN")
print("=" * 80)


# Para no generar demasiadas gráficas,
# mostramos las matrices utilizando 50/50.

for dataset in datasets.keys():

    X, y = datasets[dataset]


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=y
    )


    modelos = crear_modelos()


    for nombre_modelo, modelo in modelos.items():

        modelo.fit(
            X_train,
            y_train
        )


        y_pred = modelo.predict(
            X_test
        )


        cm = confusion_matrix(
            y_test,
            y_pred
        )


        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )


        disp.plot()


        plt.title(
            f"{dataset} - {nombre_modelo}"
        )


        plt.tight_layout()

        plt.show()


# ============================================================
# 19. RESUMEN FINAL
# ============================================================

print("\n\n")
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)


print("""
En este experimento se compararon cinco clasificadores:

1. LDA
2. K-NN con k=3
3. K-NN con k=7
4. Decision Tree
5. Decision Tree complejo

Se utilizaron cuatro bases de datos:

1. MNIST Digits
2. MNIST Fashion
3. Wine
4. Breast Cancer

Las estrategias de validación fueron:

- 50% entrenamiento / 50% prueba
- 40% entrenamiento / 60% prueba
- 5-Fold Cross Validation
- Leave-One-Out Cross Validation

Para estudiar sobreajuste se comparó el rendimiento
del entrenamiento contra el rendimiento de prueba.

Una diferencia grande entre ambos resultados puede
indicar sobreajuste.

Un rendimiento bajo tanto en entrenamiento como
en prueba puede indicar subajuste.

El modelo ganador de cada dataset se determina
utilizando el promedio de Accuracy obtenido
mediante 5-Fold Cross Validation.

IMPORTANTE:
LOOCV se ejecutó sobre una muestra de máximo 300 datos
por dataset debido al elevado costo computacional de
entrenar un modelo una vez por cada observación.
""")


print("\n")
print("=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)
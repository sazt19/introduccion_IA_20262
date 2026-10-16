# ============================================================
# CLASIFICACIÓN DEL DATASET IRIS
# Modelos: LDA, K-NN (k=3) y Decision Tree
# Partición: 70% entrenamiento / 30% prueba
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. CARGAR EL DATASET IRIS
# ============================================================

iris = load_iris()

# Seleccionamos solo dos características para visualizar
# Petal Length y Petal Width
X = iris.data[:, 2:4]
y = iris.target

feature_names = iris.feature_names[2:4]
class_names = iris.target_names

print("Características utilizadas:", feature_names)
print("Clases:", class_names)
print("Número total de muestras:", len(X))


# ============================================================
# 2. PARTICIÓN 70% ENTRENAMIENTO - 30% PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTamaño conjunto de entrenamiento:", len(X_train))
print("Tamaño conjunto de prueba:", len(X_test))


# ============================================================
# 3. NORMALIZACIÓN DE LOS DATOS
# ============================================================

# Es especialmente importante para K-NN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. DEFINICIÓN DE LOS CLASIFICADORES
# ============================================================

classifiers = {
    "LDA": LinearDiscriminantAnalysis(),

    "K-NN (k=3)": KNeighborsClassifier(
        n_neighbors=3
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}


# ============================================================
# 5. ENTRENAMIENTO Y MÉTRICAS
# ============================================================

results = {}

for name, model in classifiers.items():

    print("\n" + "=" * 60)
    print(f"MODELO: {name}")
    print("=" * 60)

    # Entrenar
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "y_pred": y_pred
    }

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names
    ))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    print("Matriz de confusión:")
    print(cm)


# ============================================================
# 6. FUNCIÓN PARA GRAFICAR FRONTERAS DE DECISIÓN
# ============================================================

def plot_decision_boundary(model, X, y, title):
    """
    Grafica las fronteras de decisión de un clasificador
    utilizando dos características.
    """

    # Límites del gráfico
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Crear una cuadrícula de puntos
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    # Predecir cada punto de la cuadrícula
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Dibujar las regiones de decisión
    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    # Dibujar los datos
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="black"
    )

    plt.xlabel("Petal Length (normalizado)")
    plt.ylabel("Petal Width (normalizado)")
    plt.title(title)

    # Leyenda
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Clases")

    plt.show()


# ============================================================
# 7. MOSTRAR LAS FRONTERAS DE DECISIÓN
# ============================================================

for name, result in results.items():

    plot_decision_boundary(
        result["model"],
        X_train_scaled,
        y_train,
        f"Fronteras de decisión - {name}"
    )


# ============================================================
# 8. MOSTRAR MATRICES DE CONFUSIÓN GRÁFICAMENTE
# ============================================================

for name, result in results.items():

    cm = confusion_matrix(y_test, result["y_pred"])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot()
    plt.title(f"Matriz de confusión - {name}")
    plt.show()


# ============================================================
# 9. COMPARACIÓN FINAL DE ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("COMPARACIÓN FINAL DE MODELOS")
print("=" * 60)

for name, result in results.items():
    print(
        f"{name}: "
        f"{result['accuracy'] * 100:.2f}%"
    )
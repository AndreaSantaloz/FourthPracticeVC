# 🧠 Aplicación con YOLO (Ultralytics)

Esta aplicación utiliza el framework **Ultralytics YOLO** para realizar tareas de visión por computador.  
A continuación se explica el flujo completo: desde la carga del modelo y el dataset, hasta su entrenamiento, validación, pruebas y exportación.

---

## 🧩 Tratamiento de datos

Primero, importamos la librería de **Ultralytics** y declaramos el modelo YOLO que vamos a utilizar.

```python
from ultralytics import YOLO

# Declaramos el modelo que vamos a usar
model = YOLO("yolo11n.yaml")
```

En caso de tener un **dataset especial** (por ejemplo, con problemas de formato o ruido), es necesario **ajustar los datos** aplicando técnicas de visión por computador o fundamentos de sistemas inteligentes.

> 📚 **Recomendación:** revisar los temas de redes neuronales de Cayetano (temas 4 al 8 o 9).  
> [Cayetano información](https://cayetanoguerra.github.io/ia/)

---

## ⚙️ Entrenamiento

Entrenamos el modelo declarando el **dataset** y el número de **épocas** (lotes de entrenamiento).

```python
results = model.train(data="coco8.yaml", epochs=5)
```

- **Dataset (`data`)**: archivo `.yaml` que describe las rutas de las imágenes y las clases.
- **Épocas (`epochs`)**: cantidad de iteraciones completas sobre el conjunto de entrenamiento.

---

## ✅ Validación

El modo **Val** se utiliza para **validar el modelo** después del entrenamiento.  
Este proceso evalúa la **precisión y capacidad de generalización** del modelo en un conjunto de validación.

```python
model.val()
```

También es posible usar un **conjunto de validación diferente**:

```python
model.val(data="path/to/separate/data.yaml")
```

### 🧮 ¿Qué es un hiperparámetro?

Un **hiperparámetro** es un valor que se define antes del entrenamiento y **controla el aprendizaje del modelo** (por ejemplo: tasa de aprendizaje, número de capas, tamaño del batch, etc.).

En fundamentos de sistemas inteligentes (FSI) se explica el concepto de **ρ(p)**, relacionado con la dinámica de ajuste de los hiperparámetros.

---

## 🔍 Pruebas (Predicción)

Después del entrenamiento y validación, realizamos **predicciones** con nuevos datos (imágenes o videos).  
El modelo detecta las **clases** y las **ubicaciones** de los objetos.

```python
# Predicción con cámara web (source=0)
results = model.predict(source="0")

# Predicción mostrando resultados en pantalla
results = model.predict(source="folder", show=True)
```

---

## 🚀 Exportación

Finalmente, el modelo se puede **exportar** para implementarlo en otros entornos.

### 🧩 Exportar a ONNX
```python
model = YOLO("yolo11n.pt")
model.export(format="onnx", dynamic=True)
```

### ⚡ Exportar a TensorRT
```python
model = YOLO("yolo11n.pt")
model.export(format="engine", device=0)
```

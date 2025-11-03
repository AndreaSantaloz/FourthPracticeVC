# Prácticas Cuatro y Cinco

###  Autores
1. Andrea Santana López


## ¿Qué se va a desarrollar?
En este entregable se va a desarrollar dos prácticas la primera consiste en un detector de matrículas utilizando YOLO y la segunda parte usaremos un conjunto de modelos propuestos que son easyocr,tesseract y paddleocr y los compararemos con sus tiempos de inferencia y rendimiento para saber cual es el mejor para la detección de matrículas.

## Condiciones para las prácticas
Para la primera práctica se tuvo que desarrollar un entorno nuevo en anaconda añadiendo con la versión en python de 3.9.5 e instalar ultralytics que es un marco de IA que ofrece implementaciones abiertas de código abierto del algoritmo de detección de objetos YOLO(You Only Live Once) y un paquete llamado lap(linear Assignment solver) es un solucionador rápido para el problema de asignación lineal(PAL). 

Aquí dejamos los pasos para que pueda ejecutar la práctica

```sh
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install ultralytics
pip install lap
```
Sino funciona lap pues use lapx
```sh
pip install lapx
```
Si cuentas con GPU, para hacer uso de su potencia de cálculo en el environment, necesitas contar con CUDA instalado, y seguir las instrucciones para contar con todos los paquetes compatibles necesarios. Para la instalación, cuentas con esta guía que te prepara la línea de comando a lanzar. Un ejemplo de la utilizada en mi equipo con CUDAv11.6
```sh
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
Luego si quieres instalar easyOcr,paddleOcr y tesseract pues tendrás que crear tres entornos virtuales ya que las dependencias de cada uno de ellos puede dejar de funcionar.
Para easyocr
<br>

```sh
pip install easyocr
```

Para tesseract  hay que ir a la página principal y descargar un archivo con extension "" y después ejecutar el siguiente comando 
<br>

```sh
pip install tesseract
```
Para paddleOcr  hay que ejecutar el siguiente comando 
<br>

```sh
pip install paddleOcr
```
## Controles del usuario
Para parar el video puede usar la teclea esc o se para solo.

## Desarrollo
Primero hay que importar las librerias para poder detectar y seguir las personas,coches y matrículas.
<br>

```python
from ultralytics import YOLO
import numpy as np
import cv2  
import math 
import os
from collections import defaultdict
from roboflow import Roboflow
```
Segundo empezamos con los apartados pedidos;en primer lugar detectamos personas y coches por lo que primero definimos el modelo que vamos a usar que es yolo11n y las clases en un diccionario donde el valor son las claves 0 y 2 y en los valores ponemos a que clase pertenecen,además declaramos el video que vamos abrir.
<br>

```python
model = YOLO("yolo11n.pt")
classes = {0:"person",2:"car"}
capture_video = cv2.VideoCapture("video1.mp4")
```

En segundo lugar realizamos un bucle while igual a true para recorrer los frames del video donde declaramos los párametros ret y frame_video donde,si ret es true significa que ha detectado un frame del video y si es true si almacena el frame en el párametro frame_video.
<br>

```python
while(True):
    ret,frame_video = capture_video.read()
```
A partir de aquí viene el tercer paso que se trata de la detección de personas y coches donde se irá explicando paso por paso.

El primer paso es generación de Resultados y preparación donde se ejecuta el modelo de detección (donde stream=True implica que los resultados se generan en tiempo real o de forma eficiente) y como resultado se crea el generador de objetos llamado results, que contiene toda la información de detección para el fotograma de video (o frame) analizado.
<br>

```python
if  ret:
        results = model(frame_video,stream=True)
```
El segundo paso es iteración por Fotograma y extracción de Cajas donde se inicia un bucle (for frames in results:) que recorre los resultados de la detección para el fotograma actual y se extraen los datos de la caja delimitadora (boxes) del resultado del fotograma, que es una lista de todos los objetos identificados y  como resultado se obtiene la colección de todos los cuadros (cajas) que rodean a los objetos detectados en el frame.
<br>

```python
 for frames in results:
            boxes = frames.boxes
```

El tercer  paso es bucle por objeto y filtrado de clase donde se inicia un segundo bucle (for box in boxes:) para procesar cada objeto individual detectado (cada box) y se comprueba el ID de clase del objeto (cls) donde se verifica si el ID de clase está dentro de las categorías de interés (if cls not in classes.keys():) sino está el comando continue que descarta este objeto y pasa al siguiente.
<br>

```python
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in classes.keys():
                continue
```
El cuarto  paso es procesamiento de datos del objeto (si es Relevante) donde si el objeto pasa el filtro (su clase es relevante), se realizan las siguientes acciones:
    1. Mostrar Clase: Se imprime en consola el nombre de la clase detectada.
    2. Coordenadas: Se extraen las coordenadas de la caja delimitadora (la ubicación del objeto).
    3. Conversión: Las coordenadas se convierten a valores enteros.

```python
         else:
            print("Clase -->",classes[cls])
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                        
```

El quinto  paso es cálculo y muestra de confianza donde se calcula el valor de confianza (el grado de seguridad del modelo sobre su detección) utilizando una fórmula de redondeo,además se aplica math.ceil((box.conf[0]*100))/100 para redondear el porcentaje de confianza a dos decimale y se obtiene el valor de confianza final del objeto para su posterior uso (por ejemplo, mostrarlo en pantalla).

```python
       confidence = math.ceil((box.conf[0]*100))/100
         print("Confianza --->",confidence)                         
```

El sexto  paso es Asignación Dinámica de Color (RGB) donde el objetivo de este código es convertir el ID de la clase detectada (cls) en un color RGB (R, G, B) para dibujar la caja delimitadora. Esto asegura que, por ejemplo, los "coches" siempre se dibujen en rojo, y las "personas" en verde, facilitando la visualización para ello se deben realizar una serie de pasos.
<br>
El primer paso cálculo de escala donde se calcula una variable escala que es un valor numérico entre 0 y 765, basándose en el ID de clase (cls) y el número total de clases (len(classes)).Esta escala representa la posición de esa clase en un espectro de color que va del rojo al azul, pasando por el amarillo y el verde.
<br>

```python
      escala = int((cls / len(classes)) * 255 * 3)                      
```
El segundo paso y último es el mapeo al Color RGB donde la serie de condicionales if/else divide la escala total (0 a 765) en tres segmentos principales (0-255, 255-510, 510-765).
<br>
Dentro de cada segmento, el código ajusta los valores R, G y B (Rojo, Verde, Azul) entre 0 y 255. Esto crea un gradiente de color continuo que garantiza que clases cercanas en ID tengan colores diferentes, pero relacionados, y que todas las clases tengan un color distinto donde se obtienen los valores finales de R, G, y B que serán usados para dibujar el cuadro.
<br>

```python
     if escala >= 255*2:
                        R = 255
                        G = 255
                        B = escala - 255*2
                    else:
                        if escala >= 255:
                            R = 255
                            G = escala - 255
                            B = 0
                        else:
                            R = escala
                            G = 0
                            B = 0                     
```
El séptimo paso y último para la deteción de objetos en el video es dibujo en el Video (OpenCV) donde una vez que se tienen las coordenadas de la caja (x1, y1, x2, y2) y el color (R, G, B), se utiliza la librería OpenCV (cv2) para pintar las detecciones en el frame del video.
<br>

```python
  cv2.rectangle(frame_video, (x1, y1), (x2, y2), (R, G, B), 3)
                    cv2.putText(frame_video, classes[cls] , [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, B), 2)
```
Donde cada función hace su trabajo en primer lugar tenemos que cv2.rectangle se encarga de crear la caja delimitadora donde le pasamos el frame_video,las coordenadas de la caja x1,y1,x2,y2,el color RGB de la clase que le corresponde y el grosor de las líneas que es 3.Con respecto a la función cv2.putText se encarga de las etiquetas donde etiqueta la caja delimitadora con la clase a la que corresponde el frame ya que le pasamos el frame del video,la clase que le corresponde usando classes[cls],aparte le pasamos las coordenadas x1 e y1 una de las esquinas de la  caja,la fuente en la que se escribirá los caracteres de la caja delimitadora,el tamaño de la fuente que usaremos que es 1,el color en el que se escribirán los carácteres de la etiqueta de la clase y por último el grosor de la fuente que le hemos establecido que es 2.

El cuarto paso del código se trata del flujo del video donde la función imshow sigue mostrando el video al pasarle como parámetros el frame del video y una etiqueta de la ventana donde se sigue emitiendose  llamada capture video  y un condicional de escape del video por si queremos cerrarlo teclemos la tecla esc.
<br>

```python
cv2.imshow('capture_video', frame_video)
if cv2.waitKey(20) == 27:
    break
```
El quinto paso y último del código es la liberación de recursos de la cámara y el cierre de todas las ventanas de opencv durante la ejecución para tener fluidez del video.
<br>

```python
# Libera el objeto de captura
capture_video.release()
# Destruye ventanas
cv2.destroyAllWindows()
```
En segundo lugar detectamos matrículas y coches que es lo mismo que lo explicado anteriormente con personas y coches pero modificando unos parámetros.
<br>

```python
classes = {0:"licenseplate",1:"car"}
model = YOLO("./runs/detect/train/weights/best.pt")
```
Donde hablaremos de estos dos parámetros el primero es classes dondeespecificamos las clases que queremos detectar qye son matrícula y coche y el segundo es 
model que es un modelo entrenado por nosotros donde, hemos cojido el dataset del modelo ya hecho de un usuario de roboflow en este siguiente link  ![modelo_robo_flow]("https://universe.roboflow.com/licenseplate-s6fjf/license-plate-xmnzu/dataset/1").

Aquí está la explicación de la descarga del dataset.
<br>

```python
rf = Roboflow(api_key="")
project = rf.workspace("licenseplate-s6fjf").project("license-plate-xmnzu")
version = project.version(1)
dataset = version.download("yolov11")
```
Aquí explicó los pasos que realiza el script para mayor entendimiento.
1. Conexión: Se usa tu clave API para establecer una conexión segura con tu cuenta de Roboflow.

```python
        rf = Roboflow(api_key="")
```
2. Identificación: Se selecciona el proyecto específico de detección de matrículas (license-plate-xmnzu) dentro de tu espacio de trabajo.
```python 
    project = rf.workspace("licenseplate-s6fjf").project("license-plate-xmnzu")
```
       
3. Versión: Se elige la Versión 1 de ese conjunto de datos.
    ```python
        version = project.version(1)

    ```
4. Descarga: Se descarga el contenido de esa versión en tu computadora, utilizando el formato yolov11 (o similar a YOLO), listo para ser usado en el entrenamiento de un modelo.
```python
        dataset = version.download("yolov11")
```

Aquí esta la explicación de como se entreno al modelo.
```python
    model = YOLO("yolo11n.pt")
    data_path = "./license-plate-1/data.yaml"
    results = model.train(data=data_path,epochs=15,imgsz=640)
```
Donde especificamos el model  con el que vamos a entrenar que es yolo11n,luego especificamos la ruta de los datos con data_path y, por último, en la variable results entrenamos el modelo con 15 épocas y dimensionando las imágenesa 640.

En tercer lugar tenemos el seguimiento de personas y coches
donde primero definimos el modelo que vamos a usar que es yolo11n y las clases en un diccionario donde el valor son las claves 0 y 2 y en los valores ponemos a que clase pertenecen,además declaramos los ids  para trackear y el video que vamos abrir.Además también hacemos un contador_clases_unicas donde inicializamos un diccionario para contar el total acumulado de objetos únicos que aparecen en el video por cada clase,también tracker_ids_contados donde inicializamos un conjunto (set) donde se almacenarán los IDs únicos de seguimiento ya contados para evitar duplicidades y establcemos el color del conteo en amarillo.Aparte inicializa un diccionario para guardar las coordenadas históricas de cada objeto, indexadas por su ID de seguimiento al que llamamos track_history.
<br>

```python
model = YOLO("yolo11n.pt")
classes = {0:"person",2:"car"}
CLASSES_TO_TRACK = [0, 2] 
vid = cv2.VideoCapture("video1.mp4")
contador_clases_unicas = {nombre: 0 for nombre in classNames.values()}
tracker_ids_contados = set()
COLOR_TEXTO_CONTEO = (0, 255, 255) # Amarillo brillante BGR
```

En segundo lugar realizamos un bucle while igual a true para recorrer los frames del video donde declaramos los párametros ret y frame_video donde,si ret es true significa que ha detectado un frame del video y si es true si almacena el frame en el párametro frame_video.
<br>

```python
while(True):
    ret,frame_video = capture_video.read()
```
A partir de aquí viene el tercer paso que se trata de la seguimiento  de personas y coches donde se irá explicando paso por paso.

El primer paso es Ejecución del Seguimiento y Asignación de IDs donde se ejecuta la función model.track(...) sobre el fotograma (img) y le añadimos los siguientes parámetros:
persist=True  que garantiza que los IDs de seguimiento se mantengan entre fotogramas,classes=CLASSES_TO_TRACK que filtra las detecciones a solo personas y coches,tracker="bytetrack.yaml" que  activa el algoritmo de seguimiento (ByteTrack) para asignar un ID único persistente a cada objeto y verbose silencia la salida de la consola durante el proceso de seguimiento.
<br>

```python
if  ret:
                results = model.track(img, persist=True, classes=CLASSES_TO_TRACK, tracker="bytetrack.yaml", verbose=False)

```
El segundo paso es Extracción de Datos y Filtrado donde se extraen las coordenadas y los IDs de los objetos seguidos haciendo lo siguiente:
1. Se verifica si hay resultados de seguimiento válidos (con IDs).
2. Se extraen las coordenadas de las cajas en formato (centro x, centro y, ancho, alto).
3. Se extraen los track_ids, que son los IDs de seguimiento únicos asignados a cada objeto.
<br>

```python
    if not results or results[0].boxes.id is None:
            pass # Si no hay detecciones o IDs de seguimiento, no hacer nada
        else:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
```

El tercer  paso es Procesamiento Individual y Conteo Único se itera sobre cada objeto detectado/seguido para extraer sus detalles y aplicar la lógica de conteo único donde se realiza en tres pasos.
1. Bucle por Objeto y Conversión de Coordenadas
Donde Se recorre cada caja (box) junto con su identificador de seguimiento (track_id) y se convierten las coordenadas de formato (centro, ancho, alto) al formato (x1, y1, x2, y2) para el dibujo de la caja.
<br>

```python

    for box, track_id in zip(boxes, track_ids):
                x1, y1, w, h = box.tolist() 
                x1, y1, x2, y2 = int(x1 - w/2), int(y1 - h/2), int(x1 + w/2), int(y1 + h/2)
       
```

2. Extracción Segura de Clase y Confianza
Aquí  se extrae la clase y la confianza del objeto, utilizando el ID de seguimiento como referencia para asegurar la correcta correspondencia.
```python
try:
    box_index = results[0].boxes.id.int().cpu().tolist().index(track_id)
    cls = int(cls_tensor[box_index])
    confidence = math.ceil((conf_tensor[box_index]*100))/100
                    clase_nombre = classNames[cls]
except ValueError:
                continue

```
Donde se utiliza un bloque try/except para encontrar el índice de la detección en los resultados original a partir del track_id, lo que permite obtener de forma segura el valor de la clase (cls), el nombre legible (clase_nombre) y la confianza (confidence).

3. Conteo de Objetos Únicos
Esta es la lógica clave para el conteo acumulado.
```python
    if track_id not in tracker_ids_contados:
                    contador_clases_unicas[clase_nombre] += 1
                    tracker_ids_contados.add(track_id)
```
Donde se realiza los siguientes pasos.
1. Se comprueba si el track_id no ha sido visto antes (no está en el set).
2. Si es nuevo: Se incrementa el contador global de esa clase ("person" o "car") en 1.
3. El track_id se añade al conjunto (tracker_ids_contados) para que nunca más se vuelva a contar.


El cuarto  paso es Asignación Dinámica de Color (RGB) donde el objetivo de este código es convertir el ID de la clase detectada (cls) en un color RGB (R, G, B) para dibujar la caja delimitadora. Esto asegura que, por ejemplo, los "coches" siempre se dibujen en rojo, y las "personas" en verde, facilitando la visualización para ello se deben realizar una serie de pasos.
<br>
El primer paso cálculo de escala donde se calcula una variable escala que es un valor numérico entre 0 y 765, basándose en el ID de clase (cls) y el número total de clases (len(classes)).Esta escala representa la posición de esa clase en un espectro de color que va del rojo al azul, pasando por el amarillo y el verde.
<br>

```python
      escala = int((cls / len(classes)) * 255 * 3)                      
```
El segundo paso y último es el mapeo al Color RGB donde la serie de condicionales if/else divide la escala total (0 a 765) en tres segmentos principales (0-255, 255-510, 510-765).
<br>
Dentro de cada segmento, el código ajusta los valores R, G y B (Rojo, Verde, Azul) entre 0 y 255. Esto crea un gradiente de color continuo que garantiza que clases cercanas en ID tengan colores diferentes, pero relacionados, y que todas las clases tengan un color distinto donde se obtienen los valores finales de R, G, y B que serán usados para dibujar el cuadro.
<br>

```python
     if escala >= 255*2:
                        R = 255
                        G = 255
                        B = escala - 255*2
                    else:
                        if escala >= 255:
                            R = 255
                            G = escala - 255
                            B = 0
                        else:
                            R = escala
                            G = 0
                            B = 0                     
```
El quinto paso y último para la deteción de objetos en el video es dibujo en el Video (OpenCV) donde una vez que se tienen las coordenadas de la caja (x1, y1, x2, y2) y el color (R, G, B), se utiliza la librería OpenCV (cv2) para pintar las detecciones en el frame del video.
<br>

```python
  cv2.rectangle(frame_video, (x1, y1), (x2, y2), (R, G, B), 3)
                    cv2.putText(frame_video, classes[cls] , [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, B), 2)
```
Donde cada función hace su trabajo en primer lugar tenemos que cv2.rectangle se encarga de crear la caja delimitadora donde le pasamos el frame_video,las coordenadas de la caja x1,y1,x2,y2,el color RGB de la clase que le corresponde y el grosor de las líneas que es 3.Con respecto a la función cv2.putText se encarga de las etiquetas donde etiqueta la caja delimitadora con la clase a la que corresponde el frame ya que le pasamos el frame del video,la clase que le corresponde usando classes[cls],aparte le pasamos las coordenadas x1 e y1 una de las esquinas de la  caja,la fuente en la que se escribirá los caracteres de la caja delimitadora,el tamaño de la fuente que usaremos que es 1,el color en el que se escribirán los carácteres de la etiqueta de la clase y por último el grosor de la fuente que le hemos establecido que es 2.
El sexto paso es Cálculo y Dibujo de Trayectoria
```python
   
                track = track_history[track_id]
                track.append((float(x1 + w/2), float(y1 + h/2))) 
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points], isClosed=False, color=(230, 230, 230), thickness=10)
```
Donde se realiza los siguientes pasos 
1. Se añade el punto central del objeto a la historia de seguimiento (track_history) indexada por su track_id.
2. Se limita la longitud de la historia a 30 puntos (track.pop(0)).
3. cv2.polylines: Dibuja el rastro o la trayectoria que el objeto ha seguido hasta el momento en el fotograma.
El séptimo paso es la visualización del conteo acumulado
<br>

```python

      for nombre, cantidad in contador_clases_unicas.items():
            texto_conteo = f"Total Unicos {nombre}: {cantidad}"
            cv2.putText(img, texto_conteo, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXTO_CONTEO, 2, cv2.LINE_AA) 
            y_offset += 30
```
Donde el cv2.putText (Conteo): Dibuja en la pantalla el conteo final acumulado de objetos únicos que han pasado.

El cuarto paso del código se trata del flujo del video donde la función imshow sigue mostrando el video al pasarle como parámetros el frame del video y una etiqueta de la ventana donde se sigue emitiendose  llamada YOLO Tracking & Unique Count  y un condicional de escape del video por si queremos cerrarlo teclemos la tecla esc.
<br>

```python
        cv2.imshow('YOLO Tracking & Unique Count', img)
        if cv2.waitKey(20) == 27:
        break
```
El quinto paso y último del código es la liberación de recursos de la cámara y el cierre de todas las ventanas de opencv durante la ejecución para tener fluidez del video.
<br>

```python
# Libera el objeto de captura
capture_video.release()
# Destruye ventanas
cv2.destroyAllWindows()
```
El sexto paso es un resumen del conteo final de las clases detectadas.
<br>

```python

print("\n--- RESUMEN FINAL DE CONTEO DE OBJETOS ÚNICOS ---")
for nombre, cantidad in contador_clases_unicas.items():
    print(f"Total acumulado de objetos únicos ({nombre}) detectados: {cantidad}")
```
El cuarto lugar hemos hecho lo mismo que lo anterior pero cambiando los parámetros model,model_path,classes y classes_to_track
<br>

```python
    MODELO_PATH = "./runs/detect/train/weights/best.pt"
    model = YOLO(MODELO_PATH) 
    classes = {0: "Matricula", 1: "Coche"}
    CLASSES_TO_TRACK = list(classes.keys()) 
```
El quinto lugar aún se esta desarrollando pero es la comparación de los modelos y la explicación de como se ha implementado easyocr y tesseract para encontrar los valores de la matrícula.También se pondrán tablas para demostrar la capacidad de ambos.

## Tencologías
1. Python 
2. Yolo

## Fuentes utilizadas

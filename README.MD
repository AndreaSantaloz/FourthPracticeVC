# Prácticas Cuatro y Cinco

###  Autores
1. Andrea Santana López
2. Luís Martín ...

## ¿Qué se va a desarrollar?
En este entregable se va a desarrollar dos prácticas la primera consiste en un detector de matrículas utilizando YOLO y la segunda .....

## Condiciones para las prácticas
Para la primera práctica se tuvo que desarrollar un entorno nuevo en anaconda añadiendo con la versión en python de 3.9.5 e instalar ultralytics que es un marco de IA que ofrece implementaciones abiertas de código abierto del algoritmo de detección de objetos YOLO(You Only Live Once) y un paquete llamado lap(linear Assignment solver) es un solucionador rápido para el problema de asignación lineal(PAL). 

Aquí dejamos los pasos para que pueda ejecutar la práctica

```
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install ultralytics
pip install lap
```
Sino funciona lap pues use lapx
```
pip install lapx
```
Si cuentas con GPU, para hacer uso de su potencia de cálculo en el environment, necesitas contar con CUDA instalado, y seguir las instrucciones para contar con todos los paquetes compatibles necesarios. Para la instalación, cuentas con esta guía que te prepara la línea de comando a lanzar. Un ejemplo de la utilizada en mi equipo con CUDAv11.6
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
## Partes importantes del detector de matrículas

## Partes importantes de la segunda práctica

## Tencologías
1. Python 
2. Yolo

## Fuentes utilizadas

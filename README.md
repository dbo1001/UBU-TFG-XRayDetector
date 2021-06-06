# Detección de defectos en piezas metálicas a partir de imágenes de radiografía  

Autora: Noelia Ubierna Fernández
Turores: D. José Francisco Díez Pastor
		 D. Pedro Latorre Carmona
Universidad de Burgos - Escuela Politécnica Superior - Grado en Ingeniería Informática

## Resumen
El control de calidad es un aspecto importante de los procesos de fabricación. Ha aumentado el nivel de competencia en el mercado, por lo que los fabricantes deben aumentar su tasa de producción manteniendo los límites de calidad para los productos.
Durante la creación de piezas metálicas se pueden crear pequeñas burbujas o poros que son indetectables por el ojo humano aun encontrándose en la superficie. Estas burbujas pueden hacer que la pieza se rompa durante su periodo de uso, pudiendo llegar a ser un problema crítico en el caso, por ejemplo, de las piezas de automóviles ya que están sometidas a una fatiga continua.
La finalidad del proyecto es la de aplicar un sistema de aprendizaje basado en una red neuronal, para la identificación de estos defectos en imágenes de rayos-X tomadas principalmente de partes automotrices (ruedas de aluminio y nudillos).


## Instalación
Comience clonando este repositorio y descargando el conjunto de datos GDXray.
A continuación, instale los paquetes necesarios:

```sh
conda env create -f environment.yml -n defect-detection
conda activate defect-detection
```

## Entrenamiento

```sh
# Python 3.6
python gdxray.py train --dataset=~/data/GDXray --series=Castings --model=mask_rcnn_coco.h5 --logs=logs/gdxray --download=True
```

## Evaluación

```sh
# Python 3.6
python gdxray.py evaluate --dataset=~/data/GDXray --series=Castings --logs=logs/gdxray --model=~/path/to/trained/model --limit=10 # Opcional el número limite de imágenes a evaluar
```

## Aplicación

```sh
# Python 3.6
python app.py 
```


## Requisitos
* Python 3.6+
* TensorFlow 1.6+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, OpenCV, Pillow, cython, h5py


## Contribuciones
* Max Ferguson: [@maxkferg](https://github.com/maxkferg)
* Stanford Engineering Informatics Group: [eil.stanford.edu](http://eil.stanford.edu/index.html)

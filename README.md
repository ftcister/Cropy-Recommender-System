# Cropy-Recommender-System

**Sistema Recomendador para Cropy**

## Instalación
Para instalar el sistema recomendador, se debe clonar el repositorio y crear un entorno virtual con python 3.11 Luego, se debe instalar las dependencias del archivo requirements.txt.

```bash
git clone
conda create -n cropy-recsys python=3.11
conda activate cropy-recsys
pip install -r requirements.txt
```

## Ejecución
Para ejecutar los modelos de Retrieval y Ranking se debe ejecutar el archivo .py correspondiente, seguido del dataset y las epochs con las flags correspondientes.

```bash
python retrieval.py --d path/to/dataset --e epochs_number
```
```bash
python ranking.py --d path/to/dataset --e epochs_number
```

## Deploy a Docker
Para hacer deploy del sistema recomendador en un contenedor de Docker, se debe crear la imagen con el archivo Dockerfile y luego ejecutar el contenedor con la imagen creada.

```bash
docker run -p -d --name recsys_retrieval 8501:8501 \                        
  --mount type=bind,source=/models/scann/model_name,target=/models/retrieval \
  -e MODEL_NAME=retrieval -t google/tf-serving-scann
```

Para hacer peticiones al sistema recomendador se puede hacer de dos formas, a traves de peticiones POST:

```bash
POST http://localhost:8501/v1/models/retrieval:predict
Content-Type: application/json

{
  "instances": [
    "User_id"
  ]
}
```

o tambien se puede hacer ejecutando el archivo predict_retrieval.py

```bash
python predict_retrieval.py -u User_id -k 10
```

Donde -u es el id del usuario y -k es el número de productos a recomendar.

Para hacer deploy al modelo de ranking, se debe crear la imagen con el archivo Dockerfile y luego ejecutar el contenedor con la imagen creada.

```bash
docker run -p -d --name recsys_ranking 8501:8501 \                        
  --mount type=bind,source=/models/ranking/model_name,target=/models/ranking \
  -e MODEL_NAME=ranking -t tensorflow/serving
```

Para hacer peticiones al sistema recomendador se puede hacer de dos formas, a traves de peticiones POST:

```bash
POST http://localhost:8501/v1/models/ranking:predict
Content-Type: application/json

{
  "instances": [
    "User_id",
    "product",
    "PRECIO",
    "sin_weekday",
    "cos_weekday",
    "sin_monthday",
    "cos_monthday",
    "sin_month",
    "cos_month",
    "sin_hour",
    "cos_hour",
  ]
}
```

o tambien se puede hacer ejecutando el archivo predict_ranking.py

```bash
python predict_ranking.py -u User_id -f product, PRECIO, sin_weekday, cos_weekday, sin_monthday, cos_monthday, sin_month, cos_month, sin_hour, cos_hour
```

Donde -u es el id del usuario y -p es el nombre de los productos a rankear.

## Dataset
Los modelos reciben como entrada datasets en formato .parquet. Estos datasets deben tener las siguientes columnas:

**Retrieval:**
- id (str): id del producto
- product (str): nombre del producto

**Ranking:**
- id (str): id del producto
- product (str): nombre del producto
- PRECIO (int): precio del producto
- sin_weekday (float): seno del día de la semana
- cos_weekday (float): coseno del día de la semana
- sin_monthday (float): seno del día del mes
- cos_monthday (float): coseno del día del mes
- sin_month (float): seno del mes
- cos_month (float): coseno del mes
- sin_hour (float): seno de la hora
- cos_hour (float): coseno de la hora

## Resultados
Los resultados de los modelos se guardan en la carpeta logs, estos resultados pueden ser analizados con TensorBoard, que por defecto corre en el puerto local :6006.

- logs/ranking/scalars/name_of_the_run: resultados del modelo de ranking
- logs/retrieval/scalars/name_of_the_run: resultados del modelo de retrieval

```bash
tensorboard --logdir logs
```
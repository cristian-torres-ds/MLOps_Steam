<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h1 align=center> Cristian Gabriel Torres </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

¡Bienvenidos al primer proyecto individual de la etapa de labs! En esta ocasión, deberán hacer un trabajo situándose en el rol de un ***MLOps Engineer***.  

<hr>  

## **Descripción del problema (Contexto y rol a desarrollar)**

## Contexto

Tienes tu modelo de recomendación dando unas buenas métricas :smirk:, y ahora, cómo lo llevas al mundo real? :eyes:

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.


## Rol a desarrollar

Empezaste a trabajar como **`Data Scientist`** en Steam, una plataforma multinacional de videojuegos. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de predecir el precio de un videojuego. :worried:

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula :sob: ): Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas….  haciendo tu trabajo imposible :weary: . 

Debes empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** (_Minimum Viable Product_) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir :exclamation:. Así que te espantas los miedos y te pones manos a la obra :muscle:

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

<sub> Nota que aqui se reflejan procesos no herramientas tecnologicas. Haz el ejercicio de entender cual herramienta del stack corresponde a cual parte del proceso<sub/>

## **Propuesta de trabajo (requerimientos de aprobación)**

**`Transformaciones`**:  Para este MVP no necesitas transformar los datos dentro del dataset pero trabajaremos en leer el dataset con el formato correcto.

**`Desarrollo API`**:   Propones disponibilizar los datos de la empresa usando el framework ***FastAPI***. Las consultas que propones son las siguientes:

Deben crear 6 funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).

+ def **genero( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los 5 géneros más vendidos en el orden correspondiente.

+ def **juegos( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los juegos lanzados en el año.

+ def **specs( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente. 

+ def **earlyacces( *`Año`: str* )**:
    Cantidad de juegos lanzados en un año con early access.

+ def **sentiment( *`Año`: str* )**:
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento. 

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *{Mixed = 182, Very Positive = 120, Positive = 278}*

+ def **metascore( *`Año`: str* )**:
    Top 5 juegos según año con mayor metascore.



<br/>


**Preparado de los datos para las consultas**<br>
Primero se cagó el dataset y se eliminaron las columnas que no se iban a usar para las consultas y se trabajó con la fecha para tratar de imputar algunas fechas que no estaban en el formato correcto. Esto se trabajó un [notebook .ipynb](https://github.com/cristian-torres-ds/henry_proyecto_individual_1/blob/main/data_seach_prep.ipynb).


**Análisis exploratorio de los datos**: _(Exploratory Data Analysis-EDA)_<br>
En [esta notebook](https://github.com/cristian-torres-ds/henry_proyecto_individual_1/blob/main/data_ml_prep.ipynb) se hizo un análisis de todas las variables, se imputaron las que se podía, se eliminaron las que se consideraban innecesarias y se trabajaron para dejar un DataFrame listo para que sea usado por el algoritmo de regresión.


**Modelo Machine Learning**: <br>
En el siguiente [notebook](https://github.com/cristian-torres-ds/henry_proyecto_individual_1/blob/main/machine_learning.ipynb) se aplicó un modelo de regresión, y se tunearon algunos hiperparámetros aplicando Grid Search, se entrenó el modelo y se lo serializó para poder usarlo en el deploy.


**Deploy**:<br/>
Una vez realizadas las [funciones](https://github.com/cristian-torres-ds/henry_proyecto_individual_1/blob/main/searchs.ipynb), se aglomeró todo en el archivo main, que es el que usamos para hacer todas las consultas y predicciones en el [deploy](https://henry-proyecto-individual-1.onrender.com).


**Video**:<br/>
Finalmente, se filmó un [video](https://youtu.be/cot87c01-tM) explicativo y demostrativo de no mas de 7 minutos sobre el trabajo realizado.


- [DEPLOY](https://henry-proyecto-individual-1.onrender.com)
- [REPOSITORIO](https://github.com/cristian-torres-ds/henry_proyecto_individual_1)
- [VIDEO](https://youtu.be/cot87c01-tM)
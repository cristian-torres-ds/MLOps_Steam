from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error


# C:\Users\cristian_torres\Desktop\HENRY\proyecto_individual_1\venv/Scripts/Activate.ps1
# para correr la app
# uvicorn main:app --reload

# para parar
# ctrl + c

# Título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Cristian Gabriel Torres DataFT13',
              description='API de datos y predicción de precios de videojuegos')



# Función para reconocer el servidor local
@app.get('/')
async def index():
    return {'Hola! Bienvenido a la API de recomedación. Por favor dirigite a /docs'}


# Dataset para búsquedas.
df_search = pd.read_json('data_search.json')


# Hacemos unos cambios en unas columnas para dejarlo listo.
df_search['release_date'] = pd.to_datetime(df_search["release_date"], errors='coerce')
df_search['metascore'] = pd.to_numeric(df_search['metascore'], errors='coerce')



# Definimos la función find_year, que vamos a usar en todas las demás funciones
def find_year(anio):
    """Función de soporte para las demas funciones, recibe un año (int)
       y devuelve un dataframe solo con los valores de ese año"""
    df_anio = df_search[df_search['release_date'].dt.year == anio]
    return df_anio



@app.get('/genero/({year})')
def genero(year:str):
    """Recibe un año y devuelve una lista con los 5 géneros
       más vendidos en el orden correspondiente. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_genero = find_year(anio)
    df_genero = df_genero.explode("genres")
    lista_generos = df_genero['genres'].value_counts().head().index.to_list()
    return {'Año' : anio, 'Generos' : lista_generos}



@app.get('/juegos/({year})')
def juegos(year):
    """Recibe un año y devuelve una lista con los juegos lanzados en el año. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_juegos = find_year(anio)
    lista_juegos = df_juegos.title.to_list()
    
    return {'Año' : anio, 'Juegos' : lista_juegos}



@app.get('/specs/({year})')
def specs(year):
    """Recibe un año y devuelve una lista con los 5 specs que 
       más se repiten en el mismo año en el orden correspondiente. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_specs = find_year(anio)
    df_specs = df_specs.explode("specs")
    lista_specs = df_specs['specs'].value_counts().head().index.to_list()
    return {'Año' : anio, 'Specs' : lista_specs}



@app.get('/earlyacces/({year})')
def earlyacces(year):
    """Recibe un año y devuelve la cantidad de juegos lanzados en ese año con early access. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_early = find_year(anio)
    early = str(df_early['early_access'].sum())

    return {'Año' : anio, 'Early acces' : early}



@app.get('/sentiment/({year})')
def sentiment(year):
    """Recibe un año y se devuelve una lista con la cantidad de registros que
       se encuentren categorizados con un análisis de sentimiento ese año. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_sentiment = find_year(anio)
    sent_on = (df_sentiment["sentiment"] == 'Overwhelmingly Negative').sum()
    sent_vn = (df_sentiment["sentiment"] == 'Very Negative').sum()
    sent_n  = (df_sentiment["sentiment"] == 'Negative').sum()
    sent_mn = (df_sentiment["sentiment"] == 'Mostly Negative').sum()
    sent_m  = (df_sentiment["sentiment"] == 'Mixed').sum()
    sent_mp = (df_sentiment["sentiment"] == 'Mostly Positive').sum()
    sent_p  = (df_sentiment["sentiment"] == 'Positive').sum()
    sent_vp = (df_sentiment["sentiment"] == 'Very Positive').sum()
    sent_op = (df_sentiment["sentiment"] == 'Overwhelmingly Positive').sum()

    sent_on_str = f"Overwhelmingly Negative: {sent_on}"
    sent_vn_str = f"Very Negative: {sent_vn}"
    sent_n_str  = f"Negative: {sent_n}"
    sent_mn_str = f"Mostly Negative: {sent_mn}"
    sent_m_str  = f"Mixed: {sent_m}"
    sent_mp_str = f"Mostly Positive: {sent_mp}"
    sent_p_str  = f"Positive: {sent_p}"
    sent_vp_str = f"Very Positive: {sent_vp}"
    sent_op_str = f"Overwhelmingly Positive: {sent_op}"

    lista = [[sent_on, sent_on_str], [sent_vn, sent_vn_str], [sent_n, sent_n_str], [sent_mn, sent_mn_str], [sent_m, sent_m_str],
             [sent_mp, sent_mp_str], [sent_p, sent_p_str], [sent_vp, sent_vp_str], [sent_op, sent_op_str]]

    lista_final = []

    for sent in lista:
        if sent[0] > 0:
            lista_final.append(sent[1])

    return {'Año' : anio, 'Sentiments' : lista_final}



@app.get('/metascore/({year})')
def metascore(year):
    """Recibe un año y retorna el top 5 juegos con mayor metascore. Ejemplo: 2017"""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_meta = find_year(anio)
    df_meta = df_meta[['title', 'metascore']].sort_values('metascore', axis=0, ascending=False).head()

    lista_name_score = []

    for i in range(df_meta.shape[0]):
        name = df_meta.iloc[i:i+1, 0:1].values[0][0]
        score = df_meta.iloc[i:i+1, 1:2].values[0][0]
        name_score = f"{name}: {score}"
        lista_name_score.append(name_score)

    return {'Año' : anio, 'Títulos' : lista_name_score}



list_sent = ["Sin reviews",
             "N user reviews",
             "Overwhelmingly negative",
             "Very negative",
             "Negative",
             "Mostly negative",
             "Mixed",
             "Mostly positive",
             "Positive",
             "Very positive",
             "Overwhelmingly positive",
            ]



@app.get("/predecir_precio")
async def predecir_precio(
    early_access: bool,
    sentiment:   str =   Query("Sin reviews", description='Sentiment', enum=list_sent),
    publisher:   str =   Query('Sin publisher', description='Publisher'),
    developer:   str =   Query('Sin developer', description='Developer'),
    year:        int =   Query(2018, description='Año'),
    month:       str =   Query('1', description='Mes', enum=['1','2','3','4','5','6','7','8','9','10','11','12']),
    genres:      list =  Query(['action','adventure'], description='Genres'),
    specs:       list =  Query(['single-player', 'multi-player'], description='Specs'),
    tags:        list =  Query(['simulation', 'strategy'], description='Tags'),
    precio_real: float = Query(0.0, description='Precio real')):


    """Esta función recibe todas las variables necesarias, y devuelve la predicción del precio.
       Si se ingresa el precio real devuelve el RMSE según la predicción, si no devuelve el mejor
       RMSE general obtenido por el modelo usando Cross Validation"""
    
    # Cargamos el modelo
    with open('lgbm_regressor_model.pkl', 'rb') as modelo:
        modelo_lgbm = pickle.load(modelo)
    
    # Cargamos tabla vacía para predicción
    with open('x_prediccion.pkl', 'rb') as x_prediccion:
        x_pred = pickle.load(x_prediccion)

    # Diccionario con publishers, ya que están separados
    # en categorías de 0 a 5 segun pupularidad
    with open('dict_publishers.pkl', 'rb') as dict_pub:
        dict_publishers = pickle.load(dict_pub)

    # Diccionario con developers, ya que están separados
    # en categorías de 0 a 5 segun pupularidad
    with open('dict_developers.pkl', 'rb') as dict_dev:
        dict_developers = pickle.load(dict_dev)


    # Early_access es True o False
    x_pred['early_access'] = early_access


    # Sentiment está compuesto por números que van del 0 al 10
    # Recibimos un string y según eso colocamos el número correspondiente
    list_sent = ["Sin reviews",
                 "N user reviews",
                 "Overwhelmingly negative",
                 "Very negative",
                 "Negative",
                 "Mostly negative",
                 "Mixed",
                 "Mostly positive",
                 "Positive",
                 "Very positive",
                 "Overwhelmingly positive",
                 ]

    for i, sent in enumerate(list_sent):
        if sent == sentiment:
            x_pred['sentiment'] = i


    # Comparamos los nombres de los publishers con respecto
    #  a las listas y les asignamos un valor numérico.
    if publisher in dict_publishers['lista_pub1']:
        x_pred['publisher_cat'] = 1
    elif publisher in dict_publishers['lista_pub2']:
        x_pred['publisher_cat'] = 2
    elif publisher in dict_publishers['lista_pub3']:
        x_pred['publisher_cat'] = 3
    elif publisher in dict_publishers['lista_pub4']:
        x_pred['publisher_cat'] = 4
    elif publisher in dict_publishers['lista_pub5']:
        x_pred['publisher_cat'] = 5
    else:
        x_pred['publisher_cat'] = 0


    # Comparamos los nombres de los developers con respecto
    #  a las listas y les asignamos un valor numérico.
    if developer in dict_developers['lista_dev1']:
        x_pred['developer_cat'] = 1
    elif developer in dict_developers['lista_dev2']:
        x_pred['developer_cat'] = 2
    elif developer in dict_developers['lista_dev3']:
        x_pred['developer_cat'] = 3
    elif developer in dict_developers['lista_dev4']:
        x_pred['developer_cat'] = 4
    elif developer in dict_developers['lista_dev5']:
        x_pred['developer_cat'] = 5
    else:
        x_pred['developer_cat'] = 0


    x_pred['year'] = year


    # Las columnas month van de 2 a 12, si el mes es cero quedan 
    # todas en 0, de otra forma se coloca un 1 en el mes correspondiente.
    if month != '1':
        columna_month = f"month_{month}"
        x_pred[columna_month] = True

    # Hacemos lista de specs, tags y genres:
    lista_features = x_pred.columns.tolist()
    lista_features = lista_features[16:]

    # Completamos con 1 las columnas correspondientes
    for genre in genres:
        if genre.lower() in lista_features:
            x_pred[genre.lower()] = 1
    for spec in specs:
        if spec.lower() in lista_features:
            x_pred[spec.lower()] = 1
    for tag in tags:
        if tag.lower() in lista_features:
            x_pred[tag.lower()] = 1


    # Hacemos la predicción
    prediccion = modelo_lgbm.predict(x_pred)

    # Hacemos este paso solo porque el método mean_squared_error necesita 
    # un valor con forma de array, aunque no sea lo más prolijo.

    if precio_real == 0:
        rmse = f"RMSE del modelo: 8.144" # MODIFICARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
        pred_str = f"Precio predicho: ${round(prediccion[0], 2)}"
        return {'prediccion': pred_str,
                'RMSE' : rmse}
    else:
        rmse = f"RMSE: {np.sqrt(mean_squared_error(pd.Series({'precio':precio_real}), prediccion))}"
        pred_str = f"Precio predicho: ${round(prediccion[0], 2)}"
        precio_real_str = f"Precio real: ${precio_real}"
        return {'prediccion': pred_str,
                'precio_real': precio_real_str,
                'RMSE' : rmse}
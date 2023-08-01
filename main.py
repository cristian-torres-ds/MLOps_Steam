from fastapi import FastAPI

# Título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Cristian Gabriel Torres DataFT13',
              description='API de datos y predicción de precios de videojuegos')

@app.get("/inicio")
async def ruta_prueba():
    return "Hola"

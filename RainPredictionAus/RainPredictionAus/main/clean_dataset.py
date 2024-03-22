import pandas as pd

# Cargar el conjunto de datos
data = pd.read_csv('../resources/weatherAUS.csv')

# Eliminar filas con valores NA o nulos
data_cleaned = data.dropna()

# Guardar el DataFrame limpio en un nuevo archivo CSV
data_cleaned.to_csv('../resources/weatherAUSClean.csv', index=False)
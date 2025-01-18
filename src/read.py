import requests
import pandas as pd


def json_to_csv(api_url, csv_filename):
    # Realiza la solicitud a la API
    response = requests.get(api_url)
    
    # Verifica que la solicitud fue exitosa
    if response.status_code == 200:
        # Carga el JSON
        data = response.json()
        
        # Convierte el JSON a un DataFrame de pandas
        df = pd.DataFrame(data)
        
        # Verifica que el DataFrame tenga las columnas esperadas
        if all(col in df.columns for col in ['id', 'temperature', 'timestamp']):
            # Guarda el DataFrame como un archivo CSV
            df.to_csv(csv_filename, index=False)
            print(f"Datos guardados en {csv_filename}")
        else:
            print("El JSON no contiene las columnas esperadas.")
    else:
        print(f"Error al realizar la solicitud: {response.status_code}")

# Ejemplo de uso
url = 'https://magicloops.dev/api/loop/f35fe175-2e71-4fad-81be-7a6b3a9aa4dc/run'
csv_filename = './data/magic_loops_data_1.csv'
json_to_csv(url, csv_filename)
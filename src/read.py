import requests
import pandas as pd
import os

def json_to_csv(api_url, base_filename):
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
            # Genera un nombre de archivo único
            csv_filename = base_filename
            counter = 1
            
            # Aumenta el nombre del archivo si ya existe
            while os.path.exists(csv_filename):
                csv_filename = f"{base_filename.rsplit('.', 1)[0]}_{counter}.csv"
                counter += 1
            
            # Guarda el DataFrame como un archivo CSV
            df.to_csv(csv_filename, index=False)
            print(f"Datos guardados en {csv_filename}")
        else:
            print("El JSON no contiene las columnas esperadas.")
    else:
        print(f"Error al realizar la solicitud: {response.status_code}")


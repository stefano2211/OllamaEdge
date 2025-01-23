import requests
import json
import time

def get_data_cripto(api_key, simbolo):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {
        'symbol': simbolo,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    while True:
        response = requests.get(url, headers=headers, params=parameters)
        if response.status_code == 200:
            datos = json.loads(response.text)
            if simbolo in datos['data']:  # Verifica que el símbolo esté en los datos
                resultado = {
                    'simbolo': simbolo,
                    'precio': datos['data'][simbolo]['quote']['USD']['price'],
                    'volumen': datos['data'][simbolo]['quote']['USD']['volume_24h'],
                    'market_cap': datos['data'][simbolo]['quote']['USD']['market_cap'],
                    'total_supply': datos['data'][simbolo]['total_supply']
                }
                return resultado
            else:
                print(f"Símbolo {simbolo} no encontrado en los datos.")
                return None
        else:
            print(f"Error en la solicitud: {response.status_code}. Reintentando en 5 segundos...")
            time.sleep(5)  # Espera 5 segundos antes de reintentar
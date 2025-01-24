import requests
import json
import time
import httpx

async def get_data_cripto(api_key, simbolo):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {
        'symbol': simbolo,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=parameters)
        if response.status_code == 200:
            datos = response.json()
            if simbolo in datos['data']:  # Check that the symbol is in the data
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
            print(f"Error en la solicitud: {response.status_code}.")
            return None
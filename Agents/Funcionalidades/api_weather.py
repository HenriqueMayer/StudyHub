from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
import requests # Solicitar um request
import datetime

# Buscando online no site: https://open-meteo.com/en/docs

# Criando uma tool (1):
class RetornTempArgs(BaseModel):
    latitude: float = Field(description='Latitude da localidade buscada.')
    longitude: float = Field(description='Longitude da localidade buscada.')


# Setup da função:
@tool(args_schema=RetornTempArgs)
def retorna_temp_atual(latitude:float, longitude:float):
    '''Retorna a temperatura atual para uma coordenada'''
    # Setup:
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m', # De quanto em quanto tempo
        'forecast_days': 1, # Quantos dias de previsão do tempo
    }

    # Request:
    resposta = requests.get(url, params=params) # Return: 200 -> Funcionou!

    if resposta.status_code == 200:
        resultado = resposta.json()

        hora_agora = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        lista_horas = [datetime.datetime.fromisoformat(temp_str) for temp_str in resultado['hourly']['time']]
        index_mais_proximo = min(range(len(lista_horas)), key=lambda x: abs(lista_horas[x] - hora_agora))

        temp_atual = resultado['hourly']['temperature_2m'][index_mais_proximo]
        
        return temp_atual
    else:
        raise Exception(f'Request para API {url} falhou. {resposta.status_code}')
    
# Testando:
retorna_temp_atual(-30,-50)


# Tool:
retorna_temp_atual.invoke({'latitude':-30, 'longitude':-50})
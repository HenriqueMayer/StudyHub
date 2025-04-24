from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
import openai

# Setup: Groq -> OpenAI
api_key = '<API_KEY>'

chat = ChatGroq(
    temperature=0, 
    groq_api_key=api_key, 
    model_name="llama3-8b-8192")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Texto que deve ser analisado:
texto = '''Henrique é um estudante de Ciência de Dados e Inteligência Artificial pela universidade PUCRS.
Ele está realizando diversos cursos e projetos para se preparar e se sentir confiante no mercado de
trabalho. Ele já fez estágio no Banrisul (2022), trabalho na Melnick no setor de Estudos Econômicos (2023-2024) e agora
está se preparando para começar um estágio na ADP Labs no time OneAI.
'''

# Função:
class InformaçãoSobreTexto(BaseModel):
    '''Informação sobre o que está acontecendo'''
    data: str = Field(description='Tentar acertar ano que está acontecendo. O formato deve ser YYYY')
    contexto: str = Field(description='Explicar o texto extraido.')

class ListaTopicosAbordados(BaseModel):
    '''Topicos para extração'''
    topicos: List[InformaçãoSobreTexto] = Field(description='Listar os assuntos e informações abordados no texto em tópios')

# Convertendo a função para uma tool:
tool_topicos = convert_to_openai_function(ListaTopicosAbordados)

# Aplicando no texto:
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Faça a extração do texto. Busque contexto, tópicos abordados, e possível timeline'),
    ('user', '{input}')
    ])

chain = (prompt
         | chat.bind(functions=[tool_topicos], function_call={'name': 'ListaTopicosAbordados'})
         | JsonOutputFunctionsParser()
)

chain.invoke({'input': texto})


# Usando uma página Web:
from langchain_community.document_loaders.web_base import  WebBaseLoader

loader = WebBaseLoader('<URL>')
page = loader.load()

# -> mesmo formato do anterior
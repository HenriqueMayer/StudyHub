# Tagging: Interpretando dados com funções
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from enum import Enum
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

# Modelo que vai informar o 'sentimento' de algum texto:
class QualSentimento(BaseModel):
    '''Define o sentimento e o idioma do texto'''
    sentimento: str = Field(description='Sentimento do texto. Informar se é "Positivo, Negativo ou Não Definido"')
    idioma: str = Field(description='Idioma do texto.')

tool_sentiemntos = convert_to_openai_function(QualSentimento)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Pense passo a passo ao categorizar o texto conforme a instrução.'),
    ('user', '{input}')
])
chain_completo = prompt | chat.bind(functions=[tool_sentiemntos], function_call={'name': 'QualSentimento'}) # Se quiser obter todas as informações
chain_apenas_resposta = (prompt
                          | chat.bind(functions=[tool_sentiemntos], function_call={'name': 'QualSentimento'})
                          | JsonOutputFunctionsParser()
                          ) # Se quiser obter apenas a resposta     

# Inserir texto na chain:
texto_1 = 'Eu amo ir na academia e treinar ombro.'
chain_completo.invoke({'input': texto_1})
chain_apenas_resposta.invoke({'input': texto_1})

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Exemplo: verificar e-mail (direcionar as pessoas para Agents específicos)
emails = [
    'Bom dia, gostaria de falar com um professor.',
    'Bom dia, gostaria de falar com o administrativo.',
    'Bom dia, gostaria de falar sobre meu boleto.',
    'Bom dia, gostaria de reclamar do meu produto, ele chegou destruido.',
    'Você ganhou um chocolate. Clique aqui para obter mais promoções',
]

class SetorEnum(str, Enum):
    colegio = 'professsores'    
    administrativo = 'administrativo e financeiro'    
    crm = 'atendimento ao cliente'    
    spam = 'spam'

class DirecionarParaSetor(BaseModel):
    '''Direciona as dúvidas ou questionamentos via email para os setores responsáveis'''
    setor: SetorEnum

tool_direcionar = convert_to_openai_function(DirecionarParaSetor)
prompt_direcionar = ChatPromptTemplate.from_messages([
    ('system', 'Você vai analisar e direcionar para os setores responsáveis as pessoas que enviaram os e-mails'), # Pode criar um 'system_message' para orientar com mais detalhe
    ('user', '{input}')
])
chain_direcionar = (prompt_direcionar
                        | chat.bind(functions=[tool_direcionar], function_call={'name': 'DirecionarParaSetor'})
                        | JsonOutputFunctionsParser()
                        )

for email in emails:
    resposta = chain_direcionar.invoke({'input': email})
    print(f'Email: {email}')
    print(resposta)
    print('---------------')
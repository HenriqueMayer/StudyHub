from langchain.agents import tool
import wikipedia

# Informar em qual o idioma que está sendo buscado:
wikipedia.set_lang('pt')

# O que estou buscando na aba de busca? criar uma query para informar:
query = 'Porto Alegre'

# Quais páginas retorna com essa query?
page_title = wikipedia.search(query)
pages_summary = []

for title in page_title:
    try:
        wiki_page = wikipedia.page(title=title, auto_suggest=True)
        pages_summary.append(f'Título da página: {title}\nResumo: {wiki_page.summary}')
    except:
        pass

pages_summary

# Após verificar que está funcionando posso atribuir uma função:
def buscar_wikipedia(query:str):
    page_title = wikipedia.search(query)
    pages_summary = []

    for title in page_title:
        try:
            wiki_page = wikipedia.page(title=title, auto_suggest=True)
            pages_summary.append(f'Título da página: {title}\nResumo: {wiki_page.summary}')
        except:
            pass
    if not pages_summary: # Se retornar vazio (caso tenha dado algum problema)
        return 'Busca não teve retorno'
    else:
        return '\n\n'.join(pages_summary)
    
print(buscar_wikipedia('Porto Alegre'))

# Criando uma tool:
@tool
def buscar_wikipedia(query:str):
    '''Faz busca no wikipedia e retorna um resumo da página'''
    page_title = wikipedia.search(query)
    pages_summary = []

    for title in page_title:
        try:
            wiki_page = wikipedia.page(title=title, auto_suggest=True)
            pages_summary.append(f'Título da página: {title}\nResumo: {wiki_page.summary}')
        except:
            pass
    if not pages_summary: # Se retornar vazio (caso tenha dado algum problema)
        return 'Busca não teve retorno'
    else:
        return '\n\n'.join(pages_summary)
    
buscar_wikipedia.invoke({'query': 'Porto Alegre'})
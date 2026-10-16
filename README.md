# Classificação de Complexidade Textual — CAEd 024/2026

Projeto desenvolvido para a prova técnica de estágio do CAEd. A aplicação classifica textos em **Ensino Fundamental I, Ensino Fundamental II, Ensino Médio ou Ensino Superior** e permite comparar a previsão antes e depois de mutações linguísticas.

O classificador utiliza **TF-IDF + canonicidade + LinearSVC**. A interface foi construída em **Streamlit** e oferece quatro operadores: **sinônimos, antônimos, gênero e paráfrase**.

## Requisitos

- **Python 3.12 (versão testada e recomendada)**. O projeto não foi validado com outras versões; o `spaCy==3.8.16` não aceita Python 3.15.
- Internet na primeira instalação, para baixar as dependências e os recursos WordNet/OMW.
- Arquivos do projeto, incluindo `requirements.txt`, `models/modelo_final.joblib`, `resources/tep/layout-two.txt` e `resources/portilexicon/pares_genero.tsv`.

## Instalação

Abra um terminal **na pasta do projeto**, onde estão `app.py` e `requirements.txt`.

### Windows (PowerShell)

**1. Instale o Python 3.12, caso ainda não esteja instalado.** Execute:

```powershell
winget install -e --id Python.Python.3.12
```

Se o comando não funcionar, instale o **Python 3.12 (Windows 64 bits)** pelo [site oficial do Python](https://www.python.org/downloads/windows/).

Depois da instalação, **feche e abra novamente o terminal** e confira:

```powershell
py -3.12 --version
```

**2. Crie o ambiente virtual e instale as dependências:**

```powershell
py -3.12 baixar_recursos.py
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader wordnet omw-2.0
```

**3. Execute a aplicação:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux ou macOS

Instale o Python 3.12 e confirme que o comando `python3.12` está disponível. Na pasta do projeto, execute:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m nltk.downloader wordnet omw-2.0
.venv/bin/python -m streamlit run app.py
```

O `requirements.txt` já inclui o modelo de português `pt_core_news_sm` utilizado pelo spaCy.

## Acesso à aplicação

Após iniciar, abra o endereço mostrado no terminal, normalmente **http://localhost:8501**.

## Como usar

1. Digite ou cole um texto na aplicação.
2. Escolha **Sinônimos**, **Antônimos**, **Gênero** ou **Paráfrase** e execute a análise.
3. Revise as sugestões e compare as classificações antes e depois das alterações.

Também é possível ativar **Comparar os quatro operadores**. Sinônimos têm modos conservador e exploratório; antônimos permitem incluir substantivos experimentalmente; gênero utiliza **PortiLexicon** com ajustes locais de concordância. Os operadores podem não sugerir alterações quando não encontram candidatos adequados às suas regras.

## Resultados e limitações

O classificador foi treinado com **2.048 textos** e avaliado em **410 textos de teste**, obtendo **acurácia de 0,9317** e **Macro-F1 de 0,9185**.

As mutações são baseadas em recursos lexicais e regras e podem gerar sugestões inadequadas ao contexto. Por isso, devem ser revisadas. A metodologia e as avaliações detalhadas estão no relatório técnico.

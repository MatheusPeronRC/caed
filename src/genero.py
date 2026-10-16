from __future__ import annotations
import unicodedata
import collections
import csv
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import spacy

MODELO_SPACY = "pt_core_news_sm"

def carregar_pares_portilexicon(
    caminho: str | Path,
    restringir_sufixos: bool = False,
) -> dict[str, tuple[str, str]]:

    por_lema = collections.defaultdict(lambda: {"Masc": set(), "Fem": set()})
    entradas = collections.defaultdict(set)

    with open(caminho, encoding="utf-8-sig", newline="") as arquivo:
        for linha in csv.reader(arquivo, delimiter="\t"):
            if len(linha) < 3:
                continue

            forma, lema, features = linha[:3]
            if not (forma.isalpha() and lema.isalpha()):
                continue

            atributos = dict(
                campo.split("=", 1)
                for campo in features.split("|")
                if "=" in campo
            )
            genero = atributos.get("Gender")
            if atributos.get("Number") != "Sing" or genero not in {"Masc", "Fem"}:
                continue

            forma, lema = forma.lower(), lema.lower()
            por_lema[lema][genero].add(forma)
            entradas[forma].add((lema, genero))

    candidatos = collections.defaultdict(set)
    for lema, generos in por_lema.items():
        homens, mulheres = generos["Masc"], generos["Fem"]
        if len(homens) != 1 or len(mulheres) != 1:
            continue
        masculino, feminino = next(iter(homens)), next(iter(mulheres))
        if masculino == feminino:
            continue

        if restringir_sufixos:
            sufixos = ("dor", "tor", "sor", "eiro", "ino", "igo")
            if not masculino.endswith(sufixos):
                continue
            if feminino != masculino + "a" and not (
                masculino.endswith(("eiro", "ino", "igo"))
                and feminino == masculino[:-1] + "a"
            ):
                continue

        if len(entradas[masculino]) != 1 or len(entradas[feminino]) != 1:
            continue
        candidatos[masculino].add((masculino, feminino))
        candidatos[feminino].add((masculino, feminino))

    return {
        forma: next(iter(pares))
        for forma, pares in candidatos.items()
        if len(pares) == 1
    }

ARQUIVO_PARES = (
    Path(__file__).resolve().parents[1]
    / "resources" / "portilexicon" / "pares_genero.tsv"
)


@lru_cache(maxsize=1)
def obter_pares_genero() -> dict[str, tuple[str, str]]:
    if not ARQUIVO_PARES.is_file():
        raise FileNotFoundError(
            "PortiLexicon não encontrado em "
            "resources/portilexicon/pares_genero.tsv. "
            "Reinstale os recursos do projeto."
        )

    resultado = {}
    with ARQUIVO_PARES.open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.reader(arquivo, delimiter="\t"):
            if len(linha) != 2:
                continue
            masculino, feminino = linha
            if not (masculino.isalpha() and feminino.isalpha()):
                continue
            par = (masculino, feminino)
            resultado[masculino] = par
            resultado[feminino] = par
    if not resultado:
        raise ValueError("O arquivo de pares do PortiLexicon está vazio.")
    return resultado


PARES_ADJETIVOS_IRREGULARES = {
    "bom": "boa",
    "mau": "má",
    "português": "portuguesa",
    "espanhol": "espanhola",
    "trabalhador": "trabalhadora",
}


PARES_DETERMINANTES = [
    ("o", "a"),
    ("os", "as"),
    ("um", "uma"),
    ("uns", "umas"),
    ("este", "esta"),
    ("estes", "estas"),
    ("esse", "essa"),
    ("esses", "essas"),
    ("aquele", "aquela"),
    ("aqueles", "aquelas"),
    ("meu", "minha"),
    ("meus", "minhas"),
    ("teu", "tua"),
    ("teus", "tuas"),
    ("seu", "sua"),
    ("seus", "suas"),
    ("nosso", "nossa"),
    ("nossos", "nossas"),
    ("vosso", "vossa"),
    ("vossos", "vossas"),
    ("algum", "alguma"),
    ("alguns", "algumas"),
    ("nenhum", "nenhuma"),
    ("nenhuns", "nenhumas"),
    ("todo", "toda"),
    ("todos", "todas"),
    ("outro", "outra"),
    ("outros", "outras"),
]


PARES_CONTRACOES = [
    ("do", "da"),
    ("dos", "das"),
    ("no", "na"),
    ("nos", "nas"),
    ("pelo", "pela"),
    ("pelos", "pelas"),
    ("ao", "à"),
    ("aos", "às"),
    ("num", "numa"),
    ("nuns", "numas"),
]

@dataclass(frozen=True)
class AlteracaoGenero:
    original: str
    novo: str

    inicio: int
    fim: int

    pos: str
    funcao: str

    @property
    def substituto(self) -> str:
        return self.novo


@dataclass(frozen=True)
class ResultadoGenero:
    texto_original: str
    texto_mutado: str

    alvo_original: str
    alvo_novo: str

    genero_original: str
    genero_novo: str

    alteracoes: list[AlteracaoGenero]
    @property
    def houve_mutacao(self) -> bool:
        return bool(self.alteracoes)

    @property
    def quantidade_alteracoes(self) -> int:
        return len(self.alteracoes)

def _criar_indice_bidirecional(pares):
    indice = {}
    for masculino, feminino in pares:
        indice[masculino] = ("Masc", feminino)
        indice[feminino] = ("Fem", masculino)

    return indice

INDICE_DETERMINANTES = _criar_indice_bidirecional(PARES_DETERMINANTES)
INDICE_CONTRACOES = _criar_indice_bidirecional(PARES_CONTRACOES)
@lru_cache(maxsize=1)
def carregar_spacy():
    return spacy.load(MODELO_SPACY)

def normalizar(texto: str) -> str:
    return unicodedata.normalize(
        "NFC",
        texto.strip().lower(),
    )

def _ajustar_caixa(
    original: str,
    candidato: str,
) -> str:
    if original.isupper():
        return candidato.upper()

    if original and original[0].isupper():
        return (
            candidato[0].upper()
            + candidato[1:]
        )

    return candidato


def _valor_morfologico(
    token,
    atributo: str,
) -> str | None:
    valores = token.morph.get(atributo)

    if not valores:
        return None
    if len(valores) != 1:
        return None

    return valores[0]


def _pluralizar_regular(palavra: str) -> str | None:
    if not palavra:
        return None

    if palavra.endswith(("a","e","i","o","u","á","é","í","ó","ú","ã","ãe","ão",)):
        return palavra + "s"

    if palavra.endswith(("r", "z")):
        return palavra + "es"

    if palavra.endswith("m"):
        return palavra[:-1] + "ns"
    return None


def _forma_com_numero(forma_singular: str,token_original,) -> str | None:
    numero = _valor_morfologico(token_original,"Number",)
    if numero != "Plur":
        return forma_singular
    return _pluralizar_regular(forma_singular)


def _encontrar_token(doc,palavra: str,inicio_alvo: int | None = None,):
    if inicio_alvo is not None:
        for token in doc:
            if token.idx == inicio_alvo:
                return token
        return None

    palavra_normalizada = normalizar(palavra)
    for token in doc:
        if normalizar(token.text) == palavra_normalizada:
            return token
    return None


def eh_alvo_genero(token, pares_externos=None) -> bool:
    if token.pos_ != "NOUN" or not token.is_alpha or token.ent_type_:
        return False

    pares = pares_externos if pares_externos is not None else obter_pares_genero()
    forma = normalizar(token.text)
    lema = normalizar(token.lemma_)
    return forma in pares or lema in pares


def _obter_forma_substantivo(token,pares_externos=None,) -> tuple[str, str, str] | None:

    if token.pos_ != "NOUN":
        return None
    if not token.is_alpha:
        return None
    if token.ent_type_:
        return None

    forma = normalizar(token.text)
    lema = normalizar(token.lemma_)
    pares = pares_externos if pares_externos is not None else obter_pares_genero()
    par = pares.get(forma)
    if par is None:
        par = pares.get(lema)
    if par is None:
        return None

    masculino, feminino = par
    genero_atual = _valor_morfologico(token,"Gender",)

    if genero_atual not in {"Masc", "Fem"}:
        if forma == masculino:
            genero_atual = "Masc"
        elif forma == feminino:
            genero_atual = "Fem"
        else:
            return None

    if genero_atual == "Masc":
        genero_novo = "Fem"
        oposto_singular = feminino
    else:
        genero_novo = "Masc"
        oposto_singular = masculino

    nova_forma = _forma_com_numero(
        oposto_singular,
        token,
    )

    if nova_forma is None:
        return None

    nova_forma = _ajustar_caixa(token.text, nova_forma,)
    return (nova_forma,genero_atual,genero_novo,)


def _flexionar_por_indice(token,indice,genero_novo: str,) -> str | None:
    forma = normalizar(token.text)
    dados = indice.get(forma)

    if dados is None:
        return None
    genero_atual, forma_oposta = dados

    if genero_atual == genero_novo:
        return token.text

    return _ajustar_caixa(token.text,forma_oposta,)


def _flexionar_adjetivo(token,genero_novo: str,) -> str | None:

    forma = normalizar(token.text)
    lema = normalizar(token.lemma_)
    numero = _valor_morfologico(token, "Number")

    masculino_irregular = None
    feminino_irregular = None

    if lema in PARES_ADJETIVOS_IRREGULARES:
        masculino_irregular = lema
        feminino_irregular = (PARES_ADJETIVOS_IRREGULARES[lema]  )
    else:
        for masculino, feminino in ( PARES_ADJETIVOS_IRREGULARES.items() ):
            if lema == feminino or forma == feminino:
                masculino_irregular = masculino
                feminino_irregular = feminino
                break

    if masculino_irregular is not None:
        base = (
            feminino_irregular
            if genero_novo == "Fem"
            else masculino_irregular
        )

        if numero == "Plur":
            base = _pluralizar_regular(base)
            if base is None:
                return None

        return _ajustar_caixa( token.text,base,)

    if lema.endswith("o"):
        if genero_novo == "Fem":
            base = lema[:-1] + "a"
        else:
            base = lema

        if numero == "Plur":
            base = _pluralizar_regular(base)
            if base is None:
                return None

        return _ajustar_caixa(token.text, base,)

    if lema.endswith("e"):
        return token.text

    genero_morfologico = _valor_morfologico(token,"Gender",)
    if genero_morfologico is None:
        return token.text

    if genero_morfologico == genero_novo:
        return token.text
    return None


def _flexionar_participio( token, genero_novo: str,) -> str | None:
    forma = normalizar(token.text)
    if genero_novo == "Fem":
        if forma.endswith("os"):
            nova_forma = forma[:-2] + "as"
        elif forma.endswith("o"):
            nova_forma = forma[:-1] + "a"
        else:
            return None

    else:
        if forma.endswith("as"):
            nova_forma = forma[:-2] + "os"
        elif forma.endswith("a"):
            nova_forma = forma[:-1] + "o"
        else:
            return None

    return _ajustar_caixa( token.text, nova_forma,)


def _coletar_dependentes(alvo,) -> list[tuple[object, str]]:
    dependentes = []
    for filho in alvo.children:
        if filho.dep_ == "det":
            dependentes.append(
                (filho, "determinante")
            )
        elif filho.dep_ == "case":
            dependentes.append(
                (filho, "contracao")
            )
        elif (
            filho.dep_ == "amod"
            and filho.pos_ == "ADJ"
        ):
            dependentes.append(
                (filho, "adjetivo")
            )

        elif (
            filho.dep_ in {"amod", "acl"}
            and "Part" in filho.morph.get("VerbForm")
        ):
            dependentes.append(
                (filho, "participio")
            )

    if (
        alvo.dep_.startswith("nsubj")
        and any(
            filho.dep_ == "cop"
            for filho in alvo.head.children
        )
    ):
        if alvo.head.pos_ == "ADJ":
            dependentes.append(
                (alvo.head, "adjetivo_predicativo")
            )
        elif "Part" in alvo.head.morph.get("VerbForm"):
            dependentes.append(
                (alvo.head, "participio_predicativo")
            )

    return dependentes


def _criar_alteracao( token, nova_forma: str,funcao: str,) -> AlteracaoGenero | None:
    if normalizar(token.text) == normalizar(nova_forma):
        return None

    return AlteracaoGenero(
        original=token.text,
        novo=nova_forma,
        inicio=token.idx,
        fim=(
            token.idx
            + len(token.text)
        ),
        pos=token.pos_,
        funcao=funcao,
    )


def _aplicar_alteracoes(texto: str,alteracoes: list[AlteracaoGenero],) -> str:
    resultado = texto

    for alteracao in sorted(
        alteracoes,
        key=lambda item: item.inicio,
        reverse=True,
    ):
        resultado = (
            resultado[:alteracao.inicio]
            + alteracao.novo
            + resultado[alteracao.fim:]
        )

    return resultado

def gerar_mutacao_genero(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
    pares_externos=None,
) -> ResultadoGenero | None:

    if not isinstance(frase, str):
        raise TypeError(
            "A frase deve ser uma string."
        )

    if not frase.strip():
        raise ValueError(
            "A frase não pode estar vazia."
        )

    if not isinstance(palavra, str):
        raise TypeError(
            "A palavra deve ser uma string."
        )

    if not palavra.strip():
        raise ValueError(
            "A palavra não pode estar vazia."
        )

    nlp = carregar_spacy()
    doc = nlp(frase)

    alvo = _encontrar_token(
        doc,
        palavra,
        inicio_alvo=inicio_alvo,
    )

    if alvo is None:
        return None

    dados_substantivo = ( _obter_forma_substantivo(alvo, pares_externos) )

    if dados_substantivo is None:
        return None

    (
        novo_substantivo,
        genero_original,
        genero_novo,
    ) = dados_substantivo

    alteracoes = []
    alteracao_alvo = _criar_alteracao(
        alvo,
        novo_substantivo,
        "substantivo",
    )
    if alteracao_alvo is not None:
        alteracoes.append(alteracao_alvo)

    for token, funcao in _coletar_dependentes(alvo):
        if funcao == "determinante":
            nova_forma = _flexionar_por_indice(
                token,
                INDICE_DETERMINANTES,
                genero_novo,
            )
            if nova_forma is None:
                genero_token = _valor_morfologico(
                    token,
                    "Gender",
                )

                if genero_token == genero_original:
                    return None

                continue

        elif funcao == "contracao":
            nova_forma = _flexionar_por_indice(
                token,
                INDICE_CONTRACOES,
                genero_novo,
            )

            if nova_forma is None:
                continue

        elif funcao in {
            "participio",
            "participio_predicativo",
        }:
            nova_forma = _flexionar_participio(
                token,
                genero_novo,
            )

            if nova_forma is None:
                return None

        else:
            nova_forma = _flexionar_adjetivo(
                token,
                genero_novo,
            )

            if nova_forma is None:
                return None

        alteracao = _criar_alteracao(
            token,
            nova_forma,
            funcao,
        )

        if alteracao is not None:
            alteracoes.append(alteracao)

    if not alteracoes:
        return None

    texto_mutado = _aplicar_alteracoes(
        frase,
        alteracoes,
    )

    return ResultadoGenero(
        texto_original=frase,
        texto_mutado=texto_mutado,
        alvo_original=alvo.text,
        alvo_novo=novo_substantivo,
        genero_original=genero_original,
        genero_novo=genero_novo,
        alteracoes=alteracoes,
    )

import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEP_PATH = (PROJECT_ROOT/ "resources"/ "tep" / "layout-two.txt")

POS_SPACY_PARA_TEP = { "NOUN": "NOUN","VERB": "VERB", "ADJ": "ADJECTIVE", "ADV": "ADVERB",}

PADRAO_LINHA = re.compile(
    r"^\[(?P<categoria>[A-Z]+)\]\s+"
    r"(?P<palavra1>.+?)\s+"
    r"(?P<relacao>SYNONYMOUS_OF|ANTONYM_OF)\s+"
    r"(?P<palavra2>.+?)$")

def normalizar(texto: str) -> str:
    return unicodedata.normalize(
        "NFC",
        texto.strip().lower() )

@lru_cache(maxsize=1)
def carregar_tep():
    if not TEP_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo do TeP não encontrado em: {TEP_PATH}"
        )

    indice = defaultdict(set)
    with TEP_PATH.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as arquivo:

        for linha in arquivo:
            linha = linha.strip()

            if not linha:
                continue

            match = PADRAO_LINHA.match(linha)

            if not match:
                continue

            if match.group("relacao") != "SYNONYMOUS_OF":
                continue

            categoria = match.group("categoria")

            palavra1 = normalizar(
                match.group("palavra1")
            )

            palavra2 = normalizar(
                match.group("palavra2")
            )

            if palavra1 == palavra2:
                continue

            indice[(categoria, palavra1)].add(
                palavra2
            )

            indice[(categoria, palavra2)].add(
                palavra1
            )

    return indice


def buscar_sinonimos(
    palavra: str,
    pos_spacy: str,
) -> list[str]:

    categoria = POS_SPACY_PARA_TEP.get(
        pos_spacy
    )

    if categoria is None:
        return []

    indice = carregar_tep()

    chave = (
        categoria,
        normalizar(palavra)
    )

    return sorted(
        indice.get(chave, set())
    )

@lru_cache(maxsize=1)
def carregar_antonimos_tep():
    if not TEP_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo do TeP não encontrado em: {TEP_PATH}"
        )

    indice = defaultdict(set)

    with TEP_PATH.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as arquivo:

        for linha in arquivo:
            linha = linha.strip()

            if not linha:
                continue

            match = PADRAO_LINHA.match(linha)

            if not match:
                continue

            if (
                match.group("relacao")
                != "ANTONYM_OF"
            ):
                continue

            categoria = match.group(
                "categoria"
            )

            palavra1 = normalizar(
                match.group("palavra1")
            )

            palavra2 = normalizar(
                match.group("palavra2")
            )

            if palavra1 == palavra2:
                continue
            indice[
                (categoria, palavra1)
            ].add(palavra2)

            indice[
                (categoria, palavra2)
            ].add(palavra1)

    return indice

def buscar_antonimos( palavra: str, pos_spacy: str,) -> list[str]:
    categoria = POS_SPACY_PARA_TEP.get(
        pos_spacy
    )

    if categoria is None:
        return []

    indice = carregar_antonimos_tep()

    chave = (
        categoria,
        normalizar(palavra)
    )

    return sorted( indice.get( chave, set() ))
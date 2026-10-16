from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
import spacy
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from .tep import buscar_sinonimos

POS_PERMITIDOS = { "NOUN","ADJ","ADV",}

FREQUENCIA_MINIMA = 2.0
DIFERENCA_ZIPF_MAXIMA = 1.5

PADRAO_PALAVRA = re.compile(
    r"^[A-Za-zÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç]+$"
)

@dataclass(frozen=True)
class CandidatoSinonimoSimples:
    original: str
    candidato: str
    lema_original: str
    pos: str
    frequencia_zipf: float
    confirmado_wordnet: bool = False
    pontuacao: float = 0.0


def pontuar_candidato(confirmado_wordnet: bool, diferenca_zipf: float) -> float:
    return (2.0 if confirmado_wordnet else 0.0) - 0.5 * diferenca_zipf


def normalizar(texto: str) -> str:
    return unicodedata.normalize(
        "NFC",
        texto.strip().lower(),
    )


@lru_cache(maxsize=1)
def carregar_spacy():
    return spacy.load("pt_core_news_sm")


def _valor_morfologico(token, atributo: str):
    valores = token.morph.get(atributo)
    if not valores:
        return None

    return valores[0]


def _encontrar_token(
    doc,
    palavra: str,
    inicio_alvo: int | None = None,
):

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


def _flexionar_adjetivo(candidato: str, alvo) -> str:
    forma = candidato

    genero = _valor_morfologico(
        alvo,
        "Gender",
    )
    numero = _valor_morfologico(
        alvo,
        "Number",
    )

    if genero == "Fem" and forma.endswith("o"):
        forma = forma[:-1] + "a"
    if numero == "Plur" and not forma.endswith("s"):
        if forma.endswith((
                "a", "e", "i", "o", "u",
                "á", "é", "í", "ó", "ú", )
        ):
            forma = forma + "s"
        elif forma.endswith(("r", "z")):
            forma = forma + "es"
        elif forma.endswith("m"):
            forma = forma[:-1] + "ns"

    return forma


def _pluralizar_substantivo(candidato: str, alvo) -> str:
    if _valor_morfologico(alvo, "Number") != "Plur":
        return candidato
    if candidato.endswith("s"):
        return candidato
    if candidato.endswith(("a", "e", "i", "o", "u")):
        return candidato + "s"
    if candidato.endswith(("r", "z")):
        return candidato + "es"
    if candidato.endswith("m"):
        return candidato[:-1] + "ns"
    return candidato


def _morfologia_compativel(original, candidato) -> bool:
    if original.pos_ != candidato.pos_:
        return False

    if original.pos_ not in {"NOUN", "ADJ"}:
        return True

    for atributo in ("Number", "Gender"):
        valor_original = _valor_morfologico(
            original,
            atributo,
        )
        valor_candidato = _valor_morfologico(
            candidato,
            atributo,
        )

        if (
            valor_original is not None
            and valor_candidato is not None
            and valor_original != valor_candidato
        ):
            return False

    return True


def _formas_wordnet(palavra: str, pos: str) -> set[str]:
    palavra = normalizar(palavra)
    formas = {palavra}

    if pos == "ADJ":
        if palavra.endswith("as"):
            formas.add(palavra[:-2] + "o")
        elif palavra.endswith("a"):
            formas.add(palavra[:-1] + "o")
        elif palavra.endswith("os"):
            formas.add(palavra[:-1])

    return formas


@lru_cache(maxsize=4096)
def _synsets_wordnet(palavra: str) -> frozenset[str]:
    return frozenset(
        synset.name()
        for synset in wn.synsets(
            palavra,
            lang="por",
        )
    )


def _wordnet_confirma_sinonimia(
    original: str,
    candidato: str,
    pos: str,
) -> bool:
    synsets_original = set()

    for forma in _formas_wordnet(original, pos):
        synsets_original.update(
            _synsets_wordnet(forma)
        )

    if not synsets_original:
        return False

    synsets_candidato = set()

    for forma in _formas_wordnet(candidato, pos):
        synsets_candidato.update(
            _synsets_wordnet(forma)
        )

    return bool(
        synsets_original
        & synsets_candidato
    )


def _buscar_candidatos_tep(alvo) -> list[str]:
    lema = normalizar(alvo.lemma_)
    forma = normalizar(alvo.text)

    candidatos = set(
        buscar_sinonimos(
            lema,
            alvo.pos_,
        )
    )

    if not candidatos and forma != lema:
        candidatos.update(
            buscar_sinonimos(
                forma,
                alvo.pos_,
            )
        )

    return sorted(candidatos)


def gerar_candidatos(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
    modo: str = "conservador",
) -> tuple[object | None, list[CandidatoSinonimoSimples]]:

    if modo not in {"conservador", "exploratorio"}:
        raise ValueError("Modo de sinonímia inválido.")

    if not isinstance(frase, str):
        raise TypeError("A frase deve ser uma string.")

    if not frase.strip():
        return None, []

    nlp = carregar_spacy()
    doc = nlp(frase)

    alvo = _encontrar_token(
        doc,
        palavra,
        inicio_alvo,
    )

    if alvo is None:
        return None, []

    if alvo.pos_ not in POS_PERMITIDOS:
        return alvo, []

    if not alvo.is_alpha or alvo.ent_type_:
        return alvo, []

    candidatos_tep = _buscar_candidatos_tep(alvo)

    if not candidatos_tep:
        return alvo, []

    original_normalizado = normalizar(alvo.text)
    lema_normalizado = normalizar(alvo.lemma_)
    frequencia_original = zipf_frequency(
        original_normalizado,
        "pt",
    )

    inicio = alvo.idx
    fim = inicio + len(alvo.text)

    candidatos_validos = []

    for candidato in candidatos_tep:
        candidato = normalizar(candidato)
        candidato_base = candidato

        if not PADRAO_PALAVRA.fullmatch(candidato):
            continue

        if candidato in {
            original_normalizado,
            lema_normalizado,
        }:
            continue

        if alvo.pos_ == "ADJ":
            candidato = _flexionar_adjetivo(
                candidato,
                alvo,
            )
        elif alvo.pos_ == "NOUN":
            candidato = _pluralizar_substantivo(candidato, alvo)

        if candidato in {
            original_normalizado,
            lema_normalizado,
        }:
            continue

        if (
            alvo.pos_ == "ADV"
            and candidato.endswith("mente") != original_normalizado.endswith("mente")
        ):
            continue

        frequencia = zipf_frequency(
            candidato,
            "pt",
        )

        frequencia_minima = 1.5 if modo == "exploratorio" else FREQUENCIA_MINIMA
        diferenca_maxima = 2.0 if modo == "exploratorio" else DIFERENCA_ZIPF_MAXIMA

        if frequencia < frequencia_minima:
            continue

        if (
            abs(frequencia - frequencia_original)
            > diferenca_maxima
        ):
            continue

        frase_candidata = (
            frase[:inicio]
            + candidato
            + frase[fim:]
        )

        doc_candidato = nlp(frase_candidata)

        token_candidato = _encontrar_token(
            doc_candidato,
            candidato,
            inicio,
        )

        if token_candidato is None:
            continue

        if not token_candidato.is_alpha:
            continue

        if not _morfologia_compativel(
            alvo,
            token_candidato,
        ):
            continue

        confirmado_wordnet = False
        if alvo.pos_ in {"ADJ", "NOUN"}:
            confirmado_wordnet = _wordnet_confirma_sinonimia(
                alvo.lemma_ if alvo.pos_ == "NOUN" else alvo.text,
                candidato_base if alvo.pos_ == "NOUN" else candidato,
                alvo.pos_,
            )
            if modo == "conservador" and not confirmado_wordnet:
                continue

        diferenca_zipf = abs(frequencia - frequencia_original)
        candidatos_validos.append(
            CandidatoSinonimoSimples(
                original=alvo.text,
                candidato=candidato,
                lema_original=lema_normalizado,
                pos=alvo.pos_,
                frequencia_zipf=frequencia,
                confirmado_wordnet=confirmado_wordnet,
                pontuacao=pontuar_candidato(
                    confirmado_wordnet, diferenca_zipf
                ),
            )
        )

    candidatos_validos.sort(
        key=lambda item: (
            -item.pontuacao,
            item.candidato,
        )
    )

    return alvo, candidatos_validos


def encontrar_melhor_sinonimo(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
    modo: str = "conservador",
) -> CandidatoSinonimoSimples | None:
    _, candidatos = gerar_candidatos(
        frase,
        palavra,
        inicio_alvo=inicio_alvo,
        modo=modo,
    )

    if not candidatos:
        return None

    return candidatos[0]


def encontrar_melhores_sinonimos(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
    modo: str = "conservador",
    limite: int = 3,
) -> list[CandidatoSinonimoSimples]:
    if not 1 <= limite <= 3:
        raise ValueError("O limite de alternativas deve estar entre 1 e 3.")
    _, candidatos = gerar_candidatos(
        frase, palavra, inicio_alvo=inicio_alvo, modo=modo
    )
    return candidatos[:limite]


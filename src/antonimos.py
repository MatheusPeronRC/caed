from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import spacy
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from .tep import buscar_antonimos

POS_PERMITIDOS = {"ADJ","ADV","NOUN",
}

# Evita candidatos muito raros
FREQUENCIA_MINIMA = 2.0


# evita trocar "certa" por "incerta" quando "errada" é uma alternativa mais frequente.
MARGEM_ZIPF_NEGACAO = 0.5

# Priorizar um candidato que já foi reconhecido como antônimo pelo wordnet 
PREFIXOS_NEGACAO = ( "des", "in", "im", "ir", "il",
)

PADRAO_PALAVRA = re.compile(
    r"^[A-Za-zÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç]+$"
)

POS_SPACY_PARA_WORDNET = {
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

@dataclass(frozen=True)
class CandidatoAntonimoSimples:
    original: str
    candidato: str
    lema_original: str
    pos: str
    frequencia_zipf: float


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

    # errado -> errada
    if genero == "Fem" and forma.endswith("o"):
        forma = forma[:-1] + "a"

    # alto -> altos / alta -> altas / grande -> grandes
    if numero == "Plur" and not forma.endswith("s"):
        if forma.endswith(
            (
                "a", "e", "i", "o", "u",
                "á", "é", "í", "ó", "ú",
            )
        ):
            forma = forma + "s"
        elif forma.endswith(("r", "z")):
            forma = forma + "es"
        elif forma.endswith("m"):
            forma = forma[:-1] + "ns"

    return forma


def _morfologia_compativel(original, candidato) -> bool:

    if original.pos_ != candidato.pos_:
        return False

    if original.pos_ == "NOUN":
        return (
            _valor_morfologico(original, "Number") == "Sing"
            and _valor_morfologico(candidato, "Number") == "Sing"
            and _valor_morfologico(original, "Gender") in {"Masc", "Fem"}
            and _valor_morfologico(original, "Gender")
            == _valor_morfologico(candidato, "Gender")
        )

    if original.pos_ != "ADJ":
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


def _formas_consulta_wordnet(alvo) -> set[str]:

    formas = {
        normalizar(alvo.lemma_),
        normalizar(alvo.text),
    }

    if alvo.pos_ == "ADJ":
        adicionais = set()

        for forma in formas:
            if forma.endswith("as"):
                adicionais.add(forma[:-2] + "o")
            elif forma.endswith("a"):
                adicionais.add(forma[:-1] + "o")
            elif forma.endswith("os"):
                adicionais.add(forma[:-1])

        formas.update(adicionais)

    return {
        forma
        for forma in formas
        if forma
    }


@lru_cache(maxsize=4096)
def _antonimos_wordnet(
    palavra: str,
    pos_spacy: str,
) -> tuple[str, ...]:

    pos_wordnet = POS_SPACY_PARA_WORDNET.get(pos_spacy)

    if pos_wordnet is None:
        return ()

    resultados = set()

    for synset in wn.synsets(
        palavra,
        lang="por",
        pos=pos_wordnet,
    ):
        for lemma in synset.lemmas():
            for antonimo in lemma.antonyms():
                synset_antonimo = antonimo.synset()

                for nome_pt in synset_antonimo.lemma_names(
                    lang="por"
                ):
                    nome_pt = normalizar(
                        nome_pt.replace("_", " ")
                    )

                    if nome_pt:
                        resultados.add(nome_pt)

    return tuple(sorted(resultados))


def _buscar_candidatos_wordnet(alvo) -> list[str]:

    candidatos = set()

    for forma in _formas_consulta_wordnet(alvo):
        candidatos.update(
            _antonimos_wordnet(
                forma,
                alvo.pos_,
            )
        )

    return sorted(candidatos)


def _eh_negacao_morfologica_direta(
    alvo,
    candidato: str,
) -> bool:


    candidato_normalizado = normalizar(candidato)

    formas_original = {
        normalizar(alvo.text),
        normalizar(alvo.lemma_),
    }

    formas_original.discard("")

    for prefixo in PREFIXOS_NEGACAO:
        if not candidato_normalizado.startswith(prefixo):
            continue

        base = candidato_normalizado[len(prefixo):]

        if base in formas_original:
            return True

    return False


def gerar_candidatos(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
) -> tuple[object | None, list[CandidatoAntonimoSimples]]:

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

    if normalizar(alvo.text) in {"melhor", "melhores"}:
        return alvo, []
    if (
        normalizar(alvo.text) == "mesmo"
        and re.search(r"\b(?:ao|no|do)\s+mesmo\s+tempo\b", frase, re.IGNORECASE)
    ):
        return alvo, []

    if alvo.pos_ == "NOUN":
        if _valor_morfologico(alvo, "Number") != "Sing":
            return alvo, []
        candidatos_lexicais = set(buscar_antonimos(alvo.text, "NOUN"))
        candidatos_lexicais.update(buscar_antonimos(alvo.lemma_, "NOUN"))
    else:
        candidatos_lexicais = _buscar_candidatos_wordnet(alvo)

    if not candidatos_lexicais:
        return alvo, []

    original_normalizado = normalizar(alvo.text)
    lema_normalizado = normalizar(alvo.lemma_)

    inicio = alvo.idx
    fim = inicio + len(alvo.text)

    candidatos_validos = []

    for candidato in candidatos_lexicais:
        candidato = normalizar(candidato)

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

        if candidato in {
            original_normalizado,
            lema_normalizado,
        }:
            continue

        frequencia = zipf_frequency(
            candidato,
            "pt",
        )

        if frequencia < FREQUENCIA_MINIMA:
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

        candidatos_validos.append(
            CandidatoAntonimoSimples(
                original=alvo.text,
                candidato=candidato,
                lema_original=lema_normalizado,
                pos=alvo.pos_,
                frequencia_zipf=frequencia,
            )
        )

    # Candidatos mais frequentes aparecem primeiro.
    # A ordem alfabética apenas desempata.
    candidatos_validos.sort(
        key=lambda item: (
            -item.frequencia_zipf,
            item.candidato,
        )
    )

    if candidatos_validos:
        melhor_frequencia = (
            candidatos_validos[0].frequencia_zipf
        )

        candidatos_negacao = [
            item
            for item in candidatos_validos
            if (
                _eh_negacao_morfologica_direta(
                    alvo,
                    item.candidato,
                )
                and (
                    melhor_frequencia
                    - item.frequencia_zipf
                ) <= MARGEM_ZIPF_NEGACAO
            )
        ]

        if candidatos_negacao:
            melhor_negacao = max(
                candidatos_negacao,
                key=lambda item: (
                    item.frequencia_zipf,
                    item.candidato,
                ),
            )

            candidatos_validos = [
                melhor_negacao,
                *(
                    item
                    for item in candidatos_validos
                    if item != melhor_negacao
                ),
            ]

    return alvo, candidatos_validos


def encontrar_melhor_antonimo(
    frase: str,
    palavra: str,
    inicio_alvo: int | None = None,
) -> CandidatoAntonimoSimples | None:
    _, candidatos = gerar_candidatos(
        frase,
        palavra,
        inicio_alvo=inicio_alvo,
    )

    if not candidatos:
        return None

    return candidatos[0]

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatoParafrase:
    original: str
    candidato: str
    padrao: str

def _capitalizar_inicio(texto: str) -> str:
    caracteres = list(texto)

    for i, caractere in enumerate(caracteres):
        if caractere.isalpha():
            caracteres[i] = caractere.upper()
            break

    return "".join(caracteres)


def _preservar_pontuacao_final(
    corpo: str,
    pontuacao: str | None,
) -> str:
    corpo = corpo.rstrip()

    if pontuacao:
        return corpo + pontuacao

    return corpo


def _resultado_diferente( original: str, candidato: str,) -> str | None:
    candidato = candidato.strip()

    if not candidato:
        return None
    if candidato == original.strip():
        return None

    return candidato

def _parafrase_quando(frase: str) -> str | None:
    match = re.fullmatch(
        r"\s*[Qq]uando\s+([^,]+),\s*(.+?)([.!?])?\s*",
        frase,
    )

    if match is None:
        return None

    subordinada, principal, pontuacao = match.groups()

    candidato = (
        f"{_capitalizar_inicio(principal.strip())} "
        f"quando {subordinada.strip()}"
    )

    return _resultado_diferente(
        frase,
        _preservar_pontuacao_final(candidato, pontuacao),
    )


def _parafrase_se(frase: str) -> str | None:
    match = re.fullmatch(
        r"\s*[Ss]e\s+([^,]+),\s*(.+?)([.!?])?\s*",
        frase,
    )

    if match is None:
        return None

    condicao, principal, pontuacao = match.groups()

    primeiro_verbo = condicao.split(maxsplit=1)[0].casefold()
    if primeiro_verbo in {
        "iniciou", "tornou", "deu", "trata", "encontra",
        "realizou", "realizaram", "fez", "faz", "apresentou",
        "registrou", "passou",
    }:
        return None

    candidato = (
        f"{_capitalizar_inicio(principal.strip())} "
        f"se {condicao.strip()}"
    )

    return _resultado_diferente(
        frase,
        _preservar_pontuacao_final(candidato, pontuacao),
    )


def _parafrase_embora(frase: str) -> str | None:
    match = re.fullmatch(
        r"\s*[Ee]mbora\s+([^,]+),\s*(.+?)([.!?])?\s*",
        frase,
    )

    if match is None:
        return None

    concessao, principal, pontuacao = match.groups()

    candidato = (
        f"{_capitalizar_inicio(principal.strip())}, "
        f"embora {concessao.strip()}"
    )

    return _resultado_diferente(
        frase,
        _preservar_pontuacao_final(candidato, pontuacao),
    )

def _parafrase_porque(frase: str) -> str | None:
    if re.search(r"\b(não\s+)?só\s+porque\b", frase, flags=re.IGNORECASE):
        return None
    if re.search(
        r"\b(?:não\s+)?(?:sei|sabe|sabemos|sabia|pergunto|perguntou)\s+porque\b",
        frase,
        flags=re.IGNORECASE,
    ):
        return None
    if len(re.findall(r"\bporque\b", frase, flags=re.IGNORECASE)) != 1:
        return None

    match = re.fullmatch(
        r"\s*(.+?)\s+porque\s+(.+?)([.!?])?\s*",
        frase,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    causa_principal, explicacao, pontuacao = match.groups()
    if explicacao.strip().rstrip(".!?").casefold() in {"sim", "não"}:
        return None

    candidato = (
        f"{causa_principal.strip()}, pois "
        f"{explicacao.strip()}"
    )

    return _resultado_diferente(
        frase,
        _preservar_pontuacao_final(candidato, pontuacao),
    )


def _parafrase_de_acordo_com(frase: str) -> str | None:
    match = re.fullmatch(
        r"\s*[Dd]e acordo com\s+([^,]+),\s*(.+?)([.!?])?\s*",
        frase,
    )

    if match is None:
        return None

    fonte, restante, pontuacao = match.groups()

    candidato = (
        f"Segundo {fonte.strip()}, "
        f"{restante.strip()}"
    )

    return _resultado_diferente(
        frase,
        _preservar_pontuacao_final(candidato, pontuacao),
    )


def _parafrase_ou_seja(frase: str) -> str | None:
    if re.search(r"\bou seja\b", frase, flags=re.IGNORECASE) is None:
        return None

    candidato = re.sub(
        r"\bou seja\b",
        lambda match: "Isto é" if match.group()[0].isupper() else "isto é",
        frase,
        count=1,
        flags=re.IGNORECASE,
    )

    return _resultado_diferente(frase, candidato)


def _parafrase_por_isso(frase: str) -> str | None:
    if re.search(r"\bpor isso\b", frase, flags=re.IGNORECASE) is None:
        return None

    candidato = re.sub(
        r"\bpor isso\b",
        lambda match: (
            "Por esse motivo" if match.group()[0].isupper()
            else "por esse motivo"
        ),
        frase,
        count=1,
        flags=re.IGNORECASE,
    )

    return _resultado_diferente(frase, candidato)

PADROES = [
    ("oracao_temporal", _parafrase_quando),
    ("oracao_condicional", _parafrase_se),
    ("oracao_concessiva", _parafrase_embora),
    ("causa_porque_pois", _parafrase_porque),
    ("fonte_de_acordo_segundo", _parafrase_de_acordo_com),
    ("explicacao_ou_seja", _parafrase_ou_seja),
    ("consequencia_por_isso", _parafrase_por_isso),
]

def gerar_candidatos(frase: str) -> list[CandidatoParafrase]:
    if not isinstance(frase, str):
        raise TypeError("A frase deve ser uma string.")

    frase = frase.strip()

    if not frase:
        raise ValueError("A frase não pode estar vazia.")

    candidatos = []
    vistos = set()

    for nome_padrao, funcao in PADROES:
        candidato = funcao(frase)

        if candidato is None:
            continue

        if candidato in vistos:
            continue

        vistos.add(candidato)

        candidatos.append(
            CandidatoParafrase(
                original=frase,
                candidato=candidato,
                padrao=nome_padrao,
            )
        )

    return candidatos


def gerar_parafrase(frase: str) -> CandidatoParafrase | None:
    candidatos = gerar_candidatos(frase)
    if not candidatos:
        return None

    return candidatos[0]

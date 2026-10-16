from __future__ import annotations

import math
import random
from dataclasses import dataclass

from . import sinonimos as ss
from . import antonimos as ant
from . import genero as gen
from . import parafrase as para

POS_SINONIMOS = { "NOUN", "ADJ", "ADV",}
POS_ANTONIMOS = { "NOUN", "ADJ","ADV",}

@dataclass(frozen=True)
class AlvoMutacao:
    palavra: str
    pos: str

    inicio: int
    fim: int

    frase: str
    inicio_na_frase: int

    inicio_frase: int
    fim_frase: int


@dataclass(frozen=True)
class AlvoParafrase:
    frase: str
    candidato: str
    padrao: str

    inicio: int
    fim: int


@dataclass(frozen=True)
class Substituicao:
    original: str
    substituto: str

    inicio: int
    fim: int

    pos: str
    frequencia_zipf: float | None = None
    funcao: str | None = None
    grupo_id: int | None = None
    alternativas: tuple[str, ...] = ()

    @property
    def sinonimo(self) -> str:
        return self.substituto

    @property
    def antonimo(self) -> str:
        return self.substituto


@dataclass
class ResultadoMutacao:
    texto_original: str
    texto_mutado: str

    substituicoes: list[Substituicao]

    palavras_elegiveis: int
    tentativas_avaliadas: int

    @property
    def quantidade_substituicoes(self) -> int:
        return len(self.substituicoes)

    @property
    def houve_mutacao(self) -> bool:
        return bool(self.substituicoes)


@dataclass
class ResultadoMutacaoGenero:
    texto_original: str
    texto_mutado: str

    substituicoes: list[Substituicao]

    palavras_elegiveis: int
    tentativas_avaliadas: int
    grupos_mutados: int

    @property
    def quantidade_substituicoes(self) -> int:
        return self.grupos_mutados

    @property
    def quantidade_alteracoes(self) -> int:
        return len(self.substituicoes)

    @property
    def houve_mutacao(self) -> bool:
        return self.grupos_mutados > 0


@dataclass
class ResultadoMutacaoParafrase:
    texto_original: str
    texto_mutado: str

    substituicoes: list[Substituicao]

    sentencas_elegiveis: int
    tentativas_avaliadas: int

    @property
    def quantidade_substituicoes(self) -> int:
        return len(self.substituicoes)

    @property
    def houve_mutacao(self) -> bool:
        return bool(self.substituicoes)

def _token_elegivel_sinonimo(token) -> bool:
    if token.pos_ not in POS_SINONIMOS:
        return False

    if not token.is_alpha:
        return False
    if token.pos_ == "ADV" and token.is_stop:
        return False

    if len(token.text) < 4:
        return False
    if token.ent_type_:
        return False

    return True


def _ajustar_caixa( original: str, candidato: str,) -> str:
    if original.isupper():
        return candidato.upper()
    if original and original[0].isupper():
        return candidato[0].upper() + candidato[1:]

    return candidato


def _frase_sem_espacos_externos(
    texto: str,
    inicio_frase: int,
    fim_frase: int,
) -> tuple[str, int, int]:
    trecho = texto[inicio_frase:fim_frase]

    esquerda = len(trecho) - len(trecho.lstrip())
    direita = len(trecho) - len(trecho.rstrip())

    inicio_ajustado = inicio_frase + esquerda
    fim_ajustado = fim_frase - direita

    return (
        texto[inicio_ajustado:fim_ajustado],
        inicio_ajustado,
        fim_ajustado,
    )


def _extrair_alvos_sinonimos( texto: str,) -> list[AlvoMutacao]:
    nlp = ss.carregar_spacy()
    doc = nlp(texto)

    alvos = []

    for sentenca in doc.sents:
        frase, inicio_frase, fim_frase = _frase_sem_espacos_externos(
            texto,
            sentenca.start_char,
            sentenca.end_char,
        )

        for token in sentenca:
            if not _token_elegivel_sinonimo(token):
                continue

            inicio = token.idx
            fim = inicio + len(token.text)

            alvos.append(
                AlvoMutacao(
                    palavra=token.text,
                    pos=token.pos_,
                    inicio=inicio,
                    fim=fim,
                    frase=frase,
                    inicio_na_frase=(
                        inicio - inicio_frase
                    ),
                    inicio_frase=inicio_frase,
                    fim_frase=fim_frase,
                )
            )

    return alvos


def _token_elegivel_antonimo(token, incluir_substantivos=False) -> bool:
    if token.pos_ not in POS_ANTONIMOS:
        return False
    if token.pos_ == "NOUN" and not incluir_substantivos:
        return False

    if not token.is_alpha:
        return False
    
    if token.pos_ == "ADV" and token.is_stop:
        return False

    if len(token.text) < 4:
        return False

    if token.ent_type_:
        return False

    return True


def _extrair_alvos_antonimos(
    texto: str,
    incluir_substantivos: bool = False,
) -> list[AlvoMutacao]:
    nlp = ant.carregar_spacy()
    doc = nlp(texto)

    alvos = []

    for sentenca in doc.sents:
        frase, inicio_frase, fim_frase = _frase_sem_espacos_externos(
            texto,
            sentenca.start_char,
            sentenca.end_char,
        )

        for token in sentenca:
            if not _token_elegivel_antonimo(token, incluir_substantivos):
                continue

            inicio = token.idx
            fim = inicio + len(token.text)

            alvos.append(
                AlvoMutacao(
                    palavra=token.text,
                    pos=token.pos_,
                    inicio=inicio,
                    fim=fim,
                    frase=frase,
                    inicio_na_frase=(
                        inicio - inicio_frase
                    ),
                    inicio_frase=inicio_frase,
                    fim_frase=fim_frase,
                )
            )

    return alvos


def _token_elegivel_genero(token, pares_externos=None) -> bool:
    return gen.eh_alvo_genero(token, pares_externos)


def _extrair_alvos_genero(
    texto: str,
    pares_externos=None,
) -> list[AlvoMutacao]:
    nlp = gen.carregar_spacy()
    doc = nlp(texto)

    alvos = []

    for sentenca in doc.sents:
        frase, inicio_frase, fim_frase = _frase_sem_espacos_externos(
            texto,
            sentenca.start_char,
            sentenca.end_char,
        )

        for token in sentenca:
            if not _token_elegivel_genero(token, pares_externos):
                continue

            inicio = token.idx
            fim = inicio + len(token.text)

            alvos.append(
                AlvoMutacao(
                    palavra=token.text,
                    pos=token.pos_,
                    inicio=inicio,
                    fim=fim,
                    frase=frase,
                    inicio_na_frase=(
                        inicio - inicio_frase
                    ),
                    inicio_frase=inicio_frase,
                    fim_frase=fim_frase,
                )
            )

    return alvos


def _extrair_alvos_parafrase( texto: str,) -> list[AlvoParafrase]:
    nlp = ss.carregar_spacy()
    doc = nlp(texto)

    alvos = []

    for sentenca in doc.sents:
        frase, inicio_frase, fim_frase = _frase_sem_espacos_externos(
            texto,
            sentenca.start_char,
            sentenca.end_char,
        )

        if not frase:
            continue

        resultado = para.gerar_parafrase(frase)
        if resultado is None:
            continue

        alvos.append(
            AlvoParafrase(
                frase=frase,
                candidato=resultado.candidato,
                padrao=resultado.padrao,
                inicio=inicio_frase,
                fim=fim_frase,
            )
        )

    return alvos


def _aplicar_substituicoes( texto: str, substituicoes: list[Substituicao],) -> str:
    resultado = texto

    for substituicao in sorted(
        substituicoes,
        key=lambda item: item.inicio,
        reverse=True,
    ):
        resultado = (
            resultado[:substituicao.inicio]
            + substituicao.substituto
            + resultado[substituicao.fim:]
        )

    return resultado


def _validar_parametros(
    texto: str,
    proporcao: float,
    max_substituicoes: int,
    max_tentativas: int,
    max_por_sentenca: int = 1,
) -> str:

    if not isinstance(texto, str):
        raise TypeError("O texto deve ser uma string.")

    texto = texto.strip()

    if not texto:
        raise ValueError("O texto não pode estar vazio.")

    if not 0 <= proporcao <= 1:
        raise ValueError("A proporção deve estar entre 0 e 1.")

    if max_substituicoes < 1:
        raise ValueError("max_substituicoes deve ser >= 1.")

    if max_tentativas < 1:
        raise ValueError("max_tentativas deve ser >= 1.")

    if max_por_sentenca < 1:
        raise ValueError("max_por_sentenca deve ser >= 1.")

    return texto

def mutar_sinonimos(
    texto: str,
    proporcao: float = 0.05,
    max_substituicoes: int = 5,
    max_tentativas: int = 20,
    seed: int = 42,
    max_por_sentenca: int = 1,
    modo: str = "conservador",
) -> ResultadoMutacao:

    texto = _validar_parametros(
        texto,
        proporcao,
        max_substituicoes,
        max_tentativas,
        max_por_sentenca,
    )
    if modo not in {"conservador", "exploratorio"}:
        raise ValueError("Modo de sinonímia inválido.")

    alvos = _extrair_alvos_sinonimos(texto)
    total_elegiveis = len(alvos)

    if total_elegiveis == 0 or proporcao == 0:
        return ResultadoMutacao(
            texto_original=texto,
            texto_mutado=texto,
            substituicoes=[],
            palavras_elegiveis=total_elegiveis,
            tentativas_avaliadas=0,
        )

    quantidade_desejada = math.ceil(
        total_elegiveis * proporcao
    )
    quantidade_desejada = max(
        1,
        quantidade_desejada,
    )
    quantidade_desejada = min(
        quantidade_desejada,
        max_substituicoes,
    )

    grupos_sentencas = {}

    for alvo in alvos:
        grupos_sentencas.setdefault(
            alvo.inicio_frase,
            [],
        ).append(alvo)

    grupos = list(grupos_sentencas.values())

    rng = random.Random(seed)
    rng.shuffle(grupos)

    for grupo in grupos:
        rng.shuffle(grupo)

    substituicoes = []
    tentativas = 0

    for grupo in grupos:
        if len(substituicoes) >= quantidade_desejada:
            break

        if tentativas >= max_tentativas:
            break

        alteracoes_na_frase = 0
        for alvo in grupo:
            if len(substituicoes) >= quantidade_desejada:
                break
            if alteracoes_na_frase >= max_por_sentenca:
                break
            if tentativas >= max_tentativas:
                break

            tentativas += 1

            candidatos = ss.encontrar_melhores_sinonimos(
                alvo.frase,
                alvo.palavra,
                inicio_alvo=alvo.inicio_na_frase,
                modo=modo,
                limite=3,
            )

            if not candidatos:
                continue

            melhor = candidatos[0]
            sinonimo = _ajustar_caixa(
                alvo.palavra,
                melhor.candidato,
            )
            alternativas = tuple(dict.fromkeys(
                _ajustar_caixa(alvo.palavra, item.candidato)
                for item in candidatos
            ))

            if ss.normalizar(sinonimo) == ss.normalizar(alvo.palavra):
                continue

            substituicoes.append(
                Substituicao(
                    original=alvo.palavra,
                    substituto=sinonimo,
                    inicio=alvo.inicio,
                    fim=alvo.fim,
                    pos=alvo.pos,
                    frequencia_zipf=melhor.frequencia_zipf,
                    alternativas=alternativas,
                )
            )

            alteracoes_na_frase += 1

    texto_mutado = _aplicar_substituicoes(
        texto,
        substituicoes,
    )

    return ResultadoMutacao(
        texto_original=texto,
        texto_mutado=texto_mutado,
        substituicoes=substituicoes,
        palavras_elegiveis=total_elegiveis,
        tentativas_avaliadas=tentativas,
    )

def mutar_antonimos(
    texto: str,
    proporcao: float = 0.05,
    max_substituicoes: int = 5,
    max_tentativas: int = 50,
    seed: int = 42,
    max_por_sentenca: int = 1,
    incluir_substantivos: bool = False,
) -> ResultadoMutacao:

    texto = _validar_parametros(
        texto,
        proporcao,
        max_substituicoes,
        max_tentativas,
        max_por_sentenca,
    )

    alvos = _extrair_alvos_antonimos(
        texto, incluir_substantivos=incluir_substantivos
    )
    total_elegiveis = len(alvos)

    if total_elegiveis == 0 or proporcao == 0:
        return ResultadoMutacao(
            texto_original=texto,
            texto_mutado=texto,
            substituicoes=[],
            palavras_elegiveis=total_elegiveis,
            tentativas_avaliadas=0,
        )

    quantidade_desejada = math.ceil(
        total_elegiveis * proporcao
    )
    quantidade_desejada = max(
        1,
        quantidade_desejada,
    )
    quantidade_desejada = min(
        quantidade_desejada,
        max_substituicoes,
    )

    grupos_sentencas = {}

    for alvo in alvos:
        grupos_sentencas.setdefault(
            alvo.inicio_frase,
            [],
        ).append(alvo)

    grupos = list(grupos_sentencas.values())

    rng = random.Random(seed)
    rng.shuffle(grupos)

    for grupo in grupos:
        rng.shuffle(grupo)

    substituicoes = []
    tentativas = 0

    for grupo in grupos:
        if len(substituicoes) >= quantidade_desejada:
            break

        if tentativas >= max_tentativas:
            break

        alteracoes_na_frase = 0
        for alvo in grupo:
            if len(substituicoes) >= quantidade_desejada:
                break
            if alteracoes_na_frase >= max_por_sentenca:
                break
            if tentativas >= max_tentativas:
                break

            tentativas += 1

            melhor = ant.encontrar_melhor_antonimo(
                alvo.frase,
                alvo.palavra,
                inicio_alvo=alvo.inicio_na_frase,
            )

            if melhor is None:
                continue

            antonimo = _ajustar_caixa(
                alvo.palavra,
                melhor.candidato,
            )

            if ant.normalizar(antonimo) == ant.normalizar(alvo.palavra):
                continue

            substituicoes.append(
                Substituicao(
                    original=alvo.palavra,
                    substituto=antonimo,
                    inicio=alvo.inicio,
                    fim=alvo.fim,
                    pos=alvo.pos,
                    frequencia_zipf=melhor.frequencia_zipf,
                )
            )

            alteracoes_na_frase += 1

    texto_mutado = _aplicar_substituicoes(
        texto,
        substituicoes,
    )

    return ResultadoMutacao(
        texto_original=texto,
        texto_mutado=texto_mutado,
        substituicoes=substituicoes,
        palavras_elegiveis=total_elegiveis,
        tentativas_avaliadas=tentativas,
    )

def mutar_genero(
    texto: str,
    proporcao: float = 0.05,
    max_substituicoes: int = 5,
    max_tentativas: int = 30,
    seed: int = 42,
    max_por_sentenca: int = 1,
    pares_externos=None,
) -> ResultadoMutacaoGenero:

    texto = _validar_parametros(
        texto,
        proporcao,
        max_substituicoes,
        max_tentativas,
        max_por_sentenca,
    )

    if pares_externos is None:
        alvos = _extrair_alvos_genero(texto)
    else:
        alvos = _extrair_alvos_genero(texto, pares_externos)
    total_elegiveis = len(alvos)

    if total_elegiveis == 0 or proporcao == 0:
        return ResultadoMutacaoGenero(
            texto_original=texto,
            texto_mutado=texto,
            substituicoes=[],
            palavras_elegiveis=total_elegiveis,
            tentativas_avaliadas=0,
            grupos_mutados=0,
        )

    quantidade_desejada = math.ceil(
        total_elegiveis * proporcao
    )
    quantidade_desejada = max(
        1,
        quantidade_desejada,
    )
    quantidade_desejada = min(
        quantidade_desejada,
        max_substituicoes,
    )

    grupos_sentencas = {}

    for alvo in alvos:
        grupos_sentencas.setdefault(
            alvo.inicio_frase,
            [],
        ).append(alvo)

    grupos = list(grupos_sentencas.values())

    rng = random.Random(seed)
    rng.shuffle(grupos)

    for grupo in grupos:
        rng.shuffle(grupo)

    substituicoes = []
    tentativas = 0
    grupos_mutados = 0

    for grupo in grupos:
        if grupos_mutados >= quantidade_desejada:
            break

        if tentativas >= max_tentativas:
            break

        alteracoes_na_frase = 0
        for alvo in grupo:
            if grupos_mutados >= quantidade_desejada:
                break
            if alteracoes_na_frase >= max_por_sentenca:
                break
            if tentativas >= max_tentativas:
                break

            tentativas += 1

            parametros_lexico = (
                {"pares_externos": pares_externos}
                if pares_externos is not None else {}
            )
            resultado = gen.gerar_mutacao_genero(
                alvo.frase,
                alvo.palavra,
                inicio_alvo=alvo.inicio_na_frase,
                **parametros_lexico,
            )

            if resultado is None:
                continue

            alteracoes_grupo = []

            for alteracao in resultado.alteracoes:
                alteracoes_grupo.append(
                    Substituicao(
                        original=alteracao.original,
                        substituto=alteracao.novo,
                        inicio=(
                            alvo.inicio_frase
                            + alteracao.inicio
                        ),
                        fim=(
                            alvo.inicio_frase
                            + alteracao.fim
                        ),
                        pos=alteracao.pos,
                        funcao=alteracao.funcao,
                        grupo_id=grupos_mutados,
                    )
                )

            if not alteracoes_grupo:
                continue

            if any(
                nova.inicio < anterior.fim and anterior.inicio < nova.fim
                for nova in alteracoes_grupo
                for anterior in substituicoes
            ):
                continue

            substituicoes.extend(alteracoes_grupo)
            grupos_mutados += 1
            alteracoes_na_frase += 1

    texto_mutado = _aplicar_substituicoes( texto,substituicoes, )

    return ResultadoMutacaoGenero(
        texto_original=texto,
        texto_mutado=texto_mutado,
        substituicoes=substituicoes,
        palavras_elegiveis=total_elegiveis,
        tentativas_avaliadas=tentativas,
        grupos_mutados=grupos_mutados,
    )

def mutar_parafrase(
    texto: str,
    proporcao: float = 0.05,
    max_substituicoes: int = 5,
    max_tentativas: int = 20,
    seed: int = 42,
) -> ResultadoMutacaoParafrase:

    texto = _validar_parametros(
        texto,
        proporcao,
        max_substituicoes,
        max_tentativas,
    )

    alvos = _extrair_alvos_parafrase(texto)
    total_elegiveis = len(alvos)

    if total_elegiveis == 0 or proporcao == 0:
        return ResultadoMutacaoParafrase(
            texto_original=texto,
            texto_mutado=texto,
            substituicoes=[],
            sentencas_elegiveis=total_elegiveis,
            tentativas_avaliadas=0,
        )

    quantidade_desejada = math.ceil(
        total_elegiveis * proporcao
    )
    quantidade_desejada = max(
        1,
        quantidade_desejada,
    )
    quantidade_desejada = min(
        quantidade_desejada,
        max_substituicoes,
    )

    rng = random.Random(seed)
    rng.shuffle(alvos)

    substituicoes = []
    tentativas = 0

    for alvo in alvos:
        if len(substituicoes) >= quantidade_desejada:
            break

        if tentativas >= max_tentativas:
            break

        tentativas += 1

        substituicoes.append(
            Substituicao(
                original=alvo.frase,
                substituto=alvo.candidato,
                inicio=alvo.inicio,
                fim=alvo.fim,
                pos="SENT",
                funcao=alvo.padrao,
            )
        )

    texto_mutado = _aplicar_substituicoes(
        texto,
        substituicoes,
    )

    return ResultadoMutacaoParafrase(
        texto_original=texto,
        texto_mutado=texto_mutado,
        substituicoes=substituicoes,
        sentencas_elegiveis=total_elegiveis,
        tentativas_avaliadas=tentativas,
    )


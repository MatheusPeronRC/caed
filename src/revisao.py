from dataclasses import replace
from html import escape

def agrupar_substituicoes(substituicoes):
    grupos = []
    posicoes = {}
    for indice, sub in enumerate(substituicoes):
        grupo_id = getattr(sub, "grupo_id", None)
        chave = ("grupo", grupo_id) if grupo_id is not None else ("item", indice)
        if chave not in posicoes:
            posicoes[chave] = len(grupos)
            grupos.append([])
        grupos[posicoes[chave]].append(indice)
    return grupos


def listar_decisoes(substituicoes, grupos_aprovados):

    grupos = agrupar_substituicoes(substituicoes)
    aprovados = set(grupos_aprovados)
    if any(type(i) is not int or i < 0 or i >= len(grupos) for i in aprovados):
        raise ValueError("Índice de grupo inválido na revisão.")

    return [
        (substituicoes[indice], grupo_indice in aprovados)
        for grupo_indice, indices in enumerate(grupos)
        for indice in indices
    ]


def aplicar_revisao(texto_original, substituicoes, grupos_aprovados):
    alteracoes = [
        substituicao
        for substituicao, mantida in listar_decisoes(substituicoes, grupos_aprovados)
        if mantida
    ]
    ordenadas = sorted(alteracoes, key=lambda sub: sub.inicio)
    fim_anterior = 0
    for sub in ordenadas:
        if not (0 <= sub.inicio < sub.fim <= len(texto_original)):
            raise ValueError("Offset inválido na revisão.")
        if sub.inicio < fim_anterior:
            raise ValueError("Há alterações sobrepostas na revisão.")
        if texto_original[sub.inicio:sub.fim] != sub.original:
            raise ValueError("A substituição não corresponde ao texto original.")
        fim_anterior = sub.fim

    texto_final = texto_original
    for sub in reversed(ordenadas):
        texto_final = (
            texto_final[:sub.inicio]
            + sub.substituto
            + texto_final[sub.fim:]
        )
    return texto_final


def escolher_alternativas(substituicoes, escolhas):
    selecionadas = list(substituicoes)
    for indice, candidato in escolhas.items():
        if type(indice) is not int or not 0 <= indice < len(selecionadas):
            raise ValueError("Índice de alternativa inválido.")
        original = selecionadas[indice]
        alternativas = getattr(original, "alternativas", ())
        if candidato not in alternativas:
            raise ValueError("Alternativa não pertence à lista gerada.")
        selecionadas[indice] = replace(original, substituto=candidato)
    return selecionadas


def previa_em_contexto(texto_original, substituicoes, destacar=False):
    if not substituicoes:
        raise ValueError("A prévia exige pelo menos uma substituição.")

    inicio_alteracao = min(sub.inicio for sub in substituicoes)
    fim_alteracao = max(sub.fim for sub in substituicoes)

    inicio = max(
        texto_original.rfind(delimitador, 0, inicio_alteracao)
        for delimitador in ".!?\n"
    ) + 1
    proximos = [
        posicao
        for delimitador in ".!?\n"
        if (posicao := texto_original.find(
            delimitador, max(inicio_alteracao, fim_alteracao - 1)
        )) != -1
    ]
    fim = min(proximos) + 1 if proximos else len(texto_original)

    trecho = texto_original[inicio:fim]
    alteracoes_locais = [
        replace(sub, inicio=sub.inicio - inicio, fim=sub.fim - inicio)
        for sub in substituicoes
    ]
    grupos = agrupar_substituicoes(alteracoes_locais)
    texto_candidato = aplicar_revisao(
        trecho, alteracoes_locais, set(range(len(grupos)))
    )
    if not destacar:
        return trecho.strip(), texto_candidato.strip()
    
    partes_antes, partes_depois = [], []
    posicao = 0
    for sub in sorted(alteracoes_locais, key=lambda item: item.inicio):
        contexto = escape(trecho[posicao:sub.inicio])
        partes_antes.append(contexto)
        partes_depois.append(contexto)
        partes_antes.append(f"<strong>{escape(sub.original)}</strong>")
        partes_depois.append(f"<strong>{escape(sub.substituto)}</strong>")
        posicao = sub.fim

    restante = escape(trecho[posicao:])
    partes_antes.append(restante)
    partes_depois.append(restante)
    return "".join(partes_antes).strip(), "".join(partes_depois).strip()

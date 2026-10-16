from __future__ import annotations
import math
from html import escape
import streamlit as st
from src.canonicidade import proporcao_canonicas

st.set_page_config(
    page_title="Classificador de Complexidade Textual",
    layout="wide",
)

@st.cache_resource
def obter_modelo():
    from src.classificador import carregar_modelo

    return carregar_modelo()


def aplicar_mutacao(
    texto: str,
    operador: str,
    proporcao: float = 0.30,
    max_substituicoes: int = 5,
    max_por_sentenca: int = 1,
    modo_sinonimos: str = "conservador",
    seed: int = 42,
    incluir_substantivos_antonimos: bool = False,
):
    from src.mutacao import (
        mutar_antonimos,
        mutar_genero,
        mutar_parafrase,
        mutar_sinonimos,
    )

    parametros = {
        "proporcao": proporcao,
        "max_substituicoes": max_substituicoes,
        "max_tentativas": 50,
        "seed": seed,
    }
    if operador == "Paráfrase":
        return mutar_parafrase(texto, **parametros)

    parametros["max_por_sentenca"] = max_por_sentenca

    if operador == "Sinônimos":
        return mutar_sinonimos(texto, modo=modo_sinonimos, **parametros)

    if operador == "Antônimos":
        return mutar_antonimos(
            texto, incluir_substantivos=incluir_substantivos_antonimos,
            **parametros
        )

    if operador == "Gênero":
        from src.genero import obter_pares_genero

        return mutar_genero(
            texto, pares_externos=obter_pares_genero(), **parametros
        )

    raise ValueError(f"Operador desconhecido: {operador}")


def comparar_operadores(
    texto: str, modelo, proporcao: float,
    max_substituicoes: int, max_por_sentenca: int, seed: int,
    incluir_substantivos_antonimos: bool = False,
):
    from src.classificador import classificar_texto

    comparacoes = {}
    for nome in ("Sinônimos", "Antônimos", "Gênero", "Paráfrase"):
        resultado = aplicar_mutacao(
            texto,
            nome,
            proporcao=proporcao,
            max_substituicoes=max_substituicoes,
            max_por_sentenca=max_por_sentenca,
            modo_sinonimos="conservador",
            incluir_substantivos_antonimos=incluir_substantivos_antonimos,
            seed=seed,
        )
        comparacoes[nome] = {
            "resultado": resultado,
            "classe": classificar_texto(resultado.texto_mutado, modelo=modelo),
            "canonicidade": proporcao_canonicas(resultado.texto_mutado),
        }
    return comparacoes



def destacar_alteracoes(texto_original, substituicoes, texto_mutado=False):
    partes = []
    inicio = 0

    for sub in sorted(substituicoes, key=lambda item: item.inicio):
        if not (inicio <= sub.inicio < sub.fim <= len(texto_original)):
            raise ValueError("Posição inválida em uma substituição.")
        if texto_original[sub.inicio:sub.fim] != sub.original:
            raise ValueError("A substituição não corresponde ao texto original.")

        partes.append(escape(texto_original[inicio:sub.inicio]))
        palavra = sub.substituto if texto_mutado else sub.original
        partes.append(f"<strong>{escape(palavra)}</strong>")
        inicio = sub.fim

    partes.append(escape(texto_original[inicio:]))
    texto_html = "".join(partes)
    return (
        '<div style="white-space: pre-wrap; line-height: 1.6;">'
        f"{texto_html}</div>"
    )


def diagnosticar_mutacao(resultado, proporcao: float, max_substituicoes: int):
    elegiveis = (
        resultado.palavras_elegiveis
        if hasattr(resultado, "palavras_elegiveis")
        else resultado.sentencas_elegiveis
    )
    meta = min(math.ceil(elegiveis * proporcao), max_substituicoes)
    return elegiveis, meta


def montar_detalhes_substituicoes(substituicoes, aprovados) -> list[dict[str, str]]:
    from src.revisao import listar_decisoes

    registros = []
    for substituicao, mantida in listar_decisoes(substituicoes, aprovados):
        registros.append(
            {
                "Original": substituicao.original,
                "Mutação": substituicao.substituto,
                "Tipo": substituicao.funcao or substituicao.pos or "-",
                "Decisão": "Mantida" if mantida else "Rejeitada",
            }
        )
    return registros

st.title("Classificador de Complexidade Textual")
st.write(
    "Classifique um texto em um nível de ensino, aplique uma mutação "
    "linguística e compare o resultado antes e depois da alteração."
)

texto = st.text_area(
    "Texto",
    height=240,
    placeholder="Digite ou cole aqui o texto que deseja analisar...",
)

modo_comparacao = st.toggle(
    "Comparar os quatro operadores",
    help="Aplica cada operador separadamente ao texto original.",
)

operador = st.selectbox(
    "Tipo de mutação",
    options=[
        "Sinônimos",
        "Antônimos",
        "Gênero",
        "Paráfrase",
    ],
    disabled=modo_comparacao,
)

if modo_comparacao:
    st.caption(
        "Os quatro operadores partem do mesmo texto. "
        "A comparação mostra as sugestões automáticas, sem revisão individual. "
    )

modo_sinonimos = "conservador"
if operador == "Sinônimos" and not modo_comparacao:
    modo_sinonimos = st.radio(
        "Modo de sinonímia",
        options=["conservador", "exploratorio"],
        format_func=lambda modo: (
            "Conservador" if modo == "conservador" else "Exploratório"
        ),
        horizontal=True,
        help="Exploratório aceita candidatos lexicais menos seguros. "
             "A revisão humana permite descartar sugestões inadequadas.",
    )

incluir_substantivos_antonimos = False
if operador == "Antônimos" or modo_comparacao:
    incluir_substantivos_antonimos = st.checkbox(
        "Incluir substantivos nos antônimos ",
        value=False,
        help="Por padrão são usados adjetivos e advérbios. "
             "O uso experimental de substantivos pode gerar sugestões "
             "menos adequadas ao contexto.",
    )

coluna_intensidade, _ = st.columns([1, 1.2], gap="medium")
with coluna_intensidade:
    with st.expander("Intensidade das mutações", expanded=False):
        proporcao_percentual = st.slider(
            "Proporção desejada dos elementos elegíveis (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="0% mantém o texto original; 100% tenta alterar todos os "
                 "alvos elegíveis, respeitando os demais limites e a abstenção.",
        )
        max_substituicoes = st.slider(
            "Máximo de mutações no texto",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
        )
        if operador == "Paráfrase" and not modo_comparacao:
            max_por_sentenca = 1
            st.caption(
                "Paráfrase: uma transformação por sentença, "
                "pois cada mutação reescreve a sentença inteira."
            )
        else:
            max_por_sentenca = st.slider(
                "Máximo de mutações por sentença",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
            )
        seed = st.number_input(
            "Semente de seleção (seed)",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="Muda a ordem das palavras ou sentenças examinadas. "
                 "Não altera o ranking dos sinônimos nem garante novas trocas.",
        )

    st.caption(
        "Padrão: 30%, até 5 mutações, 1 por sentença, seed 42. "
        "Busca tem até 50 tentativas."
    )

assinatura = (
    modo_comparacao, texto, operador, proporcao_percentual,
    max_substituicoes, max_por_sentenca, modo_sinonimos, int(seed),
    incluir_substantivos_antonimos,
)

rotulo_botao = (
    "Comparar os quatro operadores"
    if modo_comparacao else "Classificar e aplicar mutação"
)
if st.button(rotulo_botao, type="primary"):
    if not texto.strip():
        st.warning("Digite um texto antes de executar a análise.")
        st.stop()

    progresso = st.progress(0, text="0% — Iniciando análise")
    etapa = st.empty()

    try:
        etapa.caption("Carregando o classificador...")
        modelo = obter_modelo()
        from src.classificador import classificar_texto

        progresso.progress(10, text="10% — Modelo carregado")

        etapa.caption("Classificando o texto original...")
        classe_original = classificar_texto(
            texto,
            modelo=modelo,
        )
        progresso.progress(20, text="20% — Texto original classificado")

        if modo_comparacao:
            etapa.caption("Executando os quatro operadores...")
            resultados = comparar_operadores(
                texto,
                modelo,
                proporcao=proporcao_percentual / 100,
                max_substituicoes=max_substituicoes,
                max_por_sentenca=max_por_sentenca,
                seed=int(seed),
                incluir_substantivos_antonimos=incluir_substantivos_antonimos,
            )
            st.session_state["contador_comparacoes"] = (
                st.session_state.get("contador_comparacoes", 0) + 1
            )
            st.session_state["ultima_comparacao"] = {
                "id": st.session_state["contador_comparacoes"],
                "assinatura": assinatura,
                "classe_original": classe_original,
                "canonicidade_original": proporcao_canonicas(texto),
                "resultados": resultados,
            }
        else:
            etapa.caption("Buscando e aplicando mutações; esta etapa pode demorar...")
            resultado = aplicar_mutacao(
                texto,
                operador,
                proporcao=proporcao_percentual / 100,
                max_substituicoes=max_substituicoes,
                max_por_sentenca=max_por_sentenca,
                modo_sinonimos=modo_sinonimos,
                seed=int(seed),
                incluir_substantivos_antonimos=incluir_substantivos_antonimos,
            )
            progresso.progress(50, text="50% — Mutações processadas")

            etapa.caption("Classificando o texto modificado...")
            classe_mutada = classificar_texto(
                resultado.texto_mutado,
                modelo=modelo,
            )
            progresso.progress(80, text="80% — Texto modificado classificado")

            etapa.caption("Calculando canonicidade e preparando resultados...")
            canonicidade_original = proporcao_canonicas(texto)
            canonicidade_mutada = proporcao_canonicas(resultado.texto_mutado)
            st.session_state["contador_revisoes"] = (
                st.session_state.get("contador_revisoes", 0) + 1
            )
            st.session_state["ultima_analise"] = {
                "assinatura": assinatura,
                "resultado": resultado,
                "classe_original": classe_original,
                "classe_mutada": classe_mutada,
                "canonicidade_original": canonicidade_original,
                "canonicidade_mutada": canonicidade_mutada,
                "id": st.session_state["contador_revisoes"],
            }
        progresso.progress(100, text="100% — Análise concluída")
        etapa.empty()

    except FileNotFoundError as erro:
        progresso.empty()
        etapa.empty()
        st.error(str(erro))
        st.stop()
    except Exception as erro:
        progresso.empty()
        etapa.empty()
        st.error(f"Não foi possível concluir a análise: {erro}")
        st.stop()

dados = st.session_state.get("ultima_analise")
if dados and not modo_comparacao and dados["assinatura"] != assinatura:
    st.info("As configurações mudaram. Execute a análise novamente para atualizar os resultados.")

if dados and not modo_comparacao and dados["assinatura"] == assinatura:
    from src.revisao import (
        agrupar_substituicoes,
        aplicar_revisao,
        escolher_alternativas,
        previa_em_contexto,
    )
    from src.classificador import classificar_texto

    resultado = dados["resultado"]
    classe_original = dados["classe_original"]
    canonicidade_original = dados["canonicidade_original"]

    grupos = agrupar_substituicoes(resultado.substituicoes)
    aprovados = set()
    escolhas = {}
    if grupos:
        st.subheader("Revisão")
        with st.expander("Aceitar ou rejeitar substituições", expanded=True):
            for indice_grupo, indices in enumerate(grupos):
                sugestoes = [resultado.substituicoes[i] for i in indices]

                if operador == "Sinônimos" and len(indices) == 1:
                    alternativas = getattr(sugestoes[0], "alternativas", ())
                    if len(alternativas) > 1:
                        escolhido = st.selectbox(
                            f"Sinônimo para '{sugestoes[0].original}' "
                            f"(sugestão {indice_grupo + 1})",
                            options=alternativas,
                            key=f"alternativa_{dados['id']}_{indice_grupo}",
                        )
                        escolhas[indices[0]] = escolhido
                        sugestoes = escolher_alternativas(
                            sugestoes, {0: escolhido}
                        )

                descricao = "; ".join(
                    f"{item.original} → {item.substituto}"
                    for item in sugestoes
                )
                st.markdown(f"**Sugestão {indice_grupo + 1}: {descricao}**")
                frase_original, frase_candidata = previa_em_contexto(
                    resultado.texto_original, sugestoes, destacar=True
                )
                coluna_antes, coluna_depois = st.columns(2)
                with coluna_antes:
                    st.caption("Frase original")
                    st.markdown(frase_original, unsafe_allow_html=True)
                with coluna_depois:
                    st.caption("Frase com esta mutação")
                    st.markdown(frase_candidata, unsafe_allow_html=True)

                if st.checkbox(
                    "Aceitar esta sugestão",
                    value=operador != "Gênero",
                    key=f"revisao_{dados['id']}_{indice_grupo}",
                ):
                    aprovados.add(indice_grupo)

                if indice_grupo < len(grupos) - 1:
                    st.divider()

    substituicoes_escolhidas = escolher_alternativas(
        resultado.substituicoes, escolhas
    )
    texto_revisado = aplicar_revisao(
        resultado.texto_original, substituicoes_escolhidas, aprovados
    )
    if texto_revisado == resultado.texto_mutado:
        classe_mutada = dados["classe_mutada"]
        canonicidade_mutada = dados["canonicidade_mutada"]
    elif texto_revisado == resultado.texto_original:
        classe_mutada = classe_original
        canonicidade_mutada = canonicidade_original
    else:
        modelo = obter_modelo()
        classe_mutada = classificar_texto(texto_revisado, modelo=modelo)
        canonicidade_mutada = proporcao_canonicas(texto_revisado)

    st.divider()
    st.subheader("Comparação da classificação")

    coluna_original, coluna_mutada = st.columns(2)

    with coluna_original:
        st.markdown("**Texto original**")
        st.metric("Classe prevista", classe_original)
        st.metric(
            "Canonicidade",
            f"{canonicidade_original:.2%}",
        )

    with coluna_mutada:
        st.markdown("**Texto após revisão**")
        st.metric("Classe prevista", classe_mutada)
        st.metric(
            "Canonicidade",
            f"{canonicidade_mutada:.2%}",
            delta=(
                f"{(canonicidade_mutada - canonicidade_original) * 100:+.2f} p.p."
            ),
        )

    if classe_original == classe_mutada:
        st.success("A classe prevista foi mantida após a mutação.")
    else:
        st.warning(
            "A classe prevista mudou após a mutação: "
            f"{classe_original} → {classe_mutada}."
        )

    st.subheader("Texto final revisado")
    st.text_area(
        "Resultado",
        value=texto_revisado,
        height=240,
        label_visibility="collapsed",
    )

    st.subheader("Diagnóstico da mutação")
    elegiveis, meta = diagnosticar_mutacao(
        resultado,
        proporcao=proporcao_percentual / 100,
        max_substituicoes=max_substituicoes,
    )
    tipo_elegivel = (
        "Sentenças elegíveis" if operador == "Paráfrase"
        else "Palavras elegíveis"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(tipo_elegivel, elegiveis)
    col2.metric("Meta calculada", meta)
    col3.metric("Tentativas realizadas", resultado.tentativas_avaliadas)
    col4.metric("Mutações geradas", resultado.quantidade_substituicoes)
    st.metric("Mutações mantidas na revisão", len(aprovados))

    st.caption(
        "A meta depende da proporção e dos elementos elegíveis, limitada "
        "ao máximo configurado. Tentativas rejeitadas também são contadas. "
        "Para gênero, uma mutação corresponde a um grupo nominal."
    )
    if elegiveis and meta < max_substituicoes:
        st.caption(
            f"Com {elegiveis} elegíveis e {proporcao_percentual}% de proporção, "
            f"a meta foi {meta}, mesmo com teto de {max_substituicoes}."
        )
    if resultado.quantidade_substituicoes < meta:
        if resultado.tentativas_avaliadas >= 50:
            st.info(
                "O operador atingiu o limite de 50 tentativas sem alcançar "
                "a meta. Alguns candidatos podem ter sido rejeitados."
            )
        else:
            st.info(
                "A meta não foi alcançada. Pode ter faltado uma alternativa "
                "aceitável ou existido outra restrição no operador."
            )

    if resultado.houve_mutacao:
        st.info(
            f"Foram geradas {resultado.quantidade_substituicoes} "
            f"mutação(ões); {len(aprovados)} mantida(s) após revisão."
        )

        detalhes = montar_detalhes_substituicoes(
            substituicoes_escolhidas, aprovados
        )
        mantidas = [item for item in detalhes if item["Decisão"] == "Mantida"]
        rejeitadas = [item for item in detalhes if item["Decisão"] == "Rejeitada"]

        with st.expander(f"Alterações mantidas ({len(aprovados)})"):
            if mantidas:
                st.dataframe(
                    mantidas, use_container_width=True, hide_index=True
                )
            else:
                st.caption("Todas as sugestões foram rejeitadas.")

        with st.expander(f"Sugestões rejeitadas ({len(grupos) - len(aprovados)})"):
            if rejeitadas:
                st.dataframe(
                    rejeitadas, use_container_width=True, hide_index=True
                )
            else:
                st.caption("Nenhuma sugestão foi rejeitada.")
    else:
        st.info(
            "O operador se absteve: nenhuma mutação adequada foi encontrada "
            "para este texto."
        )


if modo_comparacao:
    comparacao = st.session_state.get("ultima_comparacao")
    if comparacao and comparacao["assinatura"] != assinatura:
        st.info("As configurações mudaram. Faça a comparação novamente.")

    if comparacao and comparacao["assinatura"] == assinatura:
        classe_original = comparacao["classe_original"]
        canonica_original = comparacao["canonicidade_original"]
        st.divider()
        st.subheader("Comparação dos operadores")
        st.write(f"Classe original: **{classe_original}**")
        st.write(f"Canonicidade original: **{canonica_original:.2%}**")

        linhas = []
        for nome, dados_operador in comparacao["resultados"].items():
            resultado = dados_operador["resultado"]
            classe = dados_operador["classe"]
            linhas.append({
                "Operador": nome,
                "Classe prevista": classe,
                "Classe mudou?": "Sim" if classe != classe_original else "Não",
                "Canonicidade": f"{dados_operador['canonicidade']:.2%}",
                "Mutações geradas": resultado.quantidade_substituicoes,
                "Tentativas": resultado.tentativas_avaliadas,
            })
        st.dataframe(linhas, use_container_width=True, hide_index=True)

        st.caption(
            "As mutações foram aplicadas automaticamente e de forma independente. "
        )

        for nome, dados_operador in comparacao["resultados"].items():
            resultado = dados_operador["resultado"]
            with st.expander(f"{nome} — {resultado.quantidade_substituicoes} mutação(ões)"):
                st.markdown("**Texto original**")
                st.markdown(
                    destacar_alteracoes(
                        resultado.texto_original, resultado.substituicoes
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown("**Texto gerado**")
                st.markdown(
                    destacar_alteracoes(
                        resultado.texto_original,
                        resultado.substituicoes,
                        texto_mutado=True,
                    ),
                    unsafe_allow_html=True,
                )
                if resultado.substituicoes:
                    st.dataframe(
                        [
                            {"Original": sub.original, "Substituição": sub.substituto}
                            for sub in resultado.substituicoes
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("Nenhuma alteração encontrada para este operador.")

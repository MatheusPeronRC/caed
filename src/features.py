import re
import numpy as np

from .canonicidade import (extrair_palavras, proporcao_canonicas,)

NOMES_FEATURES = [
    "num_palavras",
    "tamanho_medio_palavra",
    "palavras_por_frase",
    "proporcao_canonicas",
]

def extrair_features_linguisticas(texto: str) -> np.ndarray:
    palavras = extrair_palavras(texto)
    if not palavras:
        return np.zeros(len(NOMES_FEATURES), dtype=float)
    
    num_palavras = len(palavras)

    tamanho_medio_palavra = np.mean(
        [len(palavra) for palavra in palavras]
    )

    frases = [
        frase.strip()
        for frase in re.split(r"[.!?]+", texto)
        if frase.strip()
    ]

    num_frases = max(len(frases), 1)
    palavras_por_frase = num_palavras / num_frases
    proporcao = proporcao_canonicas(texto)

    return np.array([num_palavras,tamanho_medio_palavra,palavras_por_frase,proporcao,],dtype=float,)


def matriz_features_linguisticas(textos) -> np.ndarray:
    return np.vstack([extrair_features_linguisticas(texto) for texto in textos])

def matriz_canonicidade(textos) -> np.ndarray:
    return np.array([[proporcao_canonicas(texto)] for texto in textos ], dtype=float, )

import re
import unicodedata

VOGAIS = set("aeiouáàâãéêíóôõúü")
DIGRAFOS_CONSONANTAIS = {
    "ch": "C",
    "lh": "C",
    "nh": "C",
}

def normalizar_palavra(palavra: str) -> str:
    palavra = palavra.lower().strip()
    palavra = unicodedata.normalize("NFC", palavra)
    palavra = re.sub(
        r"[^a-záàâãéêíóôõúüç]",
        "",
        palavra
    )

    return palavra


def tokenizar_estrutura(palavra: str) -> list[str]:

    palavra = normalizar_palavra(palavra)
    if not palavra:
        return []

    estrutura = []
    i = 0
    while i < len(palavra):
        if i + 1 < len(palavra):
            par = palavra[i:i + 2]

            if par in DIGRAFOS_CONSONANTAIS:
                estrutura.append("C")
                i += 2
                continue

            if par in {"qu", "gu"}:
                if i + 2 < len(palavra) and palavra[i + 2] in "eiéêí":
                    estrutura.append("C")
                    i += 2
                    continue

        letra = palavra[i]
        if letra in VOGAIS:
            estrutura.append("V")
        else:
            estrutura.append("C")

        i += 1

    return estrutura


def eh_canonica(palavra: str) -> bool:
    estrutura = tokenizar_estrutura(palavra)

    if not estrutura:
        return False
    # Uma palavra CVCV...necessariamante possui quantidade par
    if len(estrutura) % 2 != 0:
        return False
    for i in range(0, len(estrutura), 2):
        if estrutura[i] != "C" or estrutura[i + 1] != "V":
            return False

    return True

def extrair_palavras(texto: str) -> list[str]:
    return re.findall(
        r"[A-Za-zÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç]+",
        texto
    )


def proporcao_canonicas(texto: str) -> float:
    palavras = extrair_palavras(texto)
    if not palavras:
        return 0.0
    quantidade_canonicas = sum(
        eh_canonica(palavra)
        for palavra in palavras
    )
    return quantidade_canonicas / len(palavras)
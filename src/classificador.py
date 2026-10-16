from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELO_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"


def carregar_modelo():
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODELO_PATH}"
        )
    return joblib.load(MODELO_PATH)


def classificar_texto(texto: str, modelo=None) -> str:
    if not texto or not texto.strip():
        raise ValueError("O texto não pode estar vazio.")
    if modelo is None:
        modelo = carregar_modelo()

    return modelo.predict([texto])[0]
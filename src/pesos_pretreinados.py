"""Pesos pré-treinados do GPT-2 no ``GPTModel`` — Sprint 6 (seção 5.5).

Até a Sprint 5 os pesos do modelo vinham de dois lugares: da inicialização
aleatória ou do nosso próprio treinamento em um conto de 20 mil caracteres.
Aqui eles vêm de fora: são os parâmetros que a OpenAI obteve treinando o GPT-2
em cerca de 40 GB de texto, publicados junto com o artigo do modelo.

Usar esses pesos não muda nada na arquitetura. O ``GPTModel`` da Sprint 4 já
tem a forma do GPT-2; o trabalho deste módulo é só de **correspondência**:
para cada tensor do arquivo publicado, descobrir qual parâmetro do nosso
modelo ele preenche.

Os pesos são lidos do arquivo ``model.safetensors`` do repositório
``openai-community/gpt2`` no Hugging Face Hub — os mesmos valores dos
checkpoints originais em TensorFlow que o livro usa, em um formato que é
apenas um dicionário ``nome -> tensor`` e não exige instalar o TensorFlow.
Nenhuma classe de modelo pronta é usada: o arquivo só fornece números.

Os arquivos ``*.safetensors`` não são versionados no Git (ver ``.gitignore``).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from tqdm import tqdm

__all__ = [
    "PASTA_MODELOS",
    "REPOSITORIOS_GPT2",
    "caminho_dos_pesos",
    "baixar_pesos_gpt2",
]

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "modelos"

# Repositório de cada tamanho do GPT-2 no Hugging Face Hub.
REPOSITORIOS_GPT2 = {
    "small": "openai-community/gpt2",          # 124M — 548 MB
    "medium": "openai-community/gpt2-medium",  # 355M — 1,5 GB
    "large": "openai-community/gpt2-large",    # 774M — 3,2 GB
    "xl": "openai-community/gpt2-xl",          # 1558M — 6,4 GB
}

_URL = "https://huggingface.co/{repositorio}/resolve/main/model.safetensors"


# ---------------------------------------------------------------------------
# 1. Download do arquivo de pesos
# ---------------------------------------------------------------------------
def caminho_dos_pesos(tamanho: str = "small", pasta: Path | str = PASTA_MODELOS) -> Path:
    """Caminho local do arquivo de pesos de um tamanho do GPT-2."""
    if tamanho not in REPOSITORIOS_GPT2:
        raise ValueError(
            f"tamanho desconhecido: {tamanho!r}. Opções: {sorted(REPOSITORIOS_GPT2)}."
        )
    return Path(pasta) / f"gpt2-{tamanho}.safetensors"


def baixar_pesos_gpt2(
    tamanho: str = "small", pasta: Path | str = PASTA_MODELOS, mostrar_progresso: bool = True
) -> Path:
    """Baixa os pesos do GPT-2 uma única vez e devolve o caminho do arquivo.

    Se o arquivo já existe em ``pasta``, nada é baixado. O download é gravado
    primeiro em um arquivo ``.part`` e só renomeado ao terminar, de modo que
    uma interrupção não deixa para trás um arquivo incompleto com o nome final.
    """
    destino = caminho_dos_pesos(tamanho, pasta)
    if destino.exists():
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    url = _URL.format(repositorio=REPOSITORIOS_GPT2[tamanho])
    requisicao = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})

    with urllib.request.urlopen(requisicao) as resposta, open(parcial, "wb") as arquivo:
        total = int(resposta.headers.get("Content-Length", 0)) or None
        with tqdm(
            total=total, unit="B", unit_scale=True, desc=destino.name,
            disable=not mostrar_progresso,
        ) as barra:
            while bloco := resposta.read(1 << 20):
                arquivo.write(bloco)
                barra.update(len(bloco))

    if total is not None and parcial.stat().st_size != total:
        parcial.unlink()
        raise OSError(f"download incompleto de {url}")
    parcial.replace(destino)
    return destino


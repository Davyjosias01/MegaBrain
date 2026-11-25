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

Correspondência entre o arquivo e o ``GPTModel`` (``i`` é o índice do bloco)::

    arquivo (GPT-2)            GPTModel                              operação
    -------------------------  ------------------------------------  ----------
    wte.weight                 embedding.token_emb.weight            cópia
    wpe.weight                 embedding.pos_emb.weight              cópia
    h.i.ln_1.weight / bias     trf_blocks.i.norm1.scale / shift      cópia
    h.i.attn.c_attn.weight     trf_blocks.i.att.W_query.weight       fatia 0, transposta
                               trf_blocks.i.att.W_key.weight         fatia 1, transposta
                               trf_blocks.i.att.W_value.weight       fatia 2, transposta
    h.i.attn.c_attn.bias       trf_blocks.i.att.W_{query,key,value}.bias   fatias 0, 1, 2
    h.i.attn.c_proj.weight     trf_blocks.i.att.out_proj.weight      transposta
    h.i.attn.c_proj.bias       trf_blocks.i.att.out_proj.bias        cópia
    h.i.ln_2.weight / bias     trf_blocks.i.norm2.scale / shift      cópia
    h.i.mlp.c_fc.weight        trf_blocks.i.ff.layers.0.weight       transposta
    h.i.mlp.c_fc.bias          trf_blocks.i.ff.layers.0.bias         cópia
    h.i.mlp.c_proj.weight      trf_blocks.i.ff.layers.2.weight       transposta
    h.i.mlp.c_proj.bias        trf_blocks.i.ff.layers.2.bias         cópia
    ln_f.weight / bias         final_norm.scale / shift              cópia
    wte.weight                 out_head.weight                       cópia

Três diferenças de formato explicam as operações:

- **Transposição.** O GPT-2 original implementa as camadas lineares com uma
  classe ``Conv1D`` que guarda o peso como ``(entrada, saída)``. O
  ``nn.Linear`` do PyTorch guarda ``(saída, entrada)``. É a mesma matriz,
  transposta.
- **Fatias de ``c_attn``.** No arquivo, as projeções de *query*, *key* e
  *value* são uma única camada com saída ``3·d``. Na nossa
  ``MultiHeadAttention`` são três ``nn.Linear`` separadas. As colunas
  ``[0, d)``, ``[d, 2d)`` e ``[2d, 3d)`` correspondem a Q, K e V, nessa ordem.
- **Cabeça de saída.** O GPT-2 não tem uma matriz própria para ``out_head``:
  reutiliza a tabela de embeddings de token (*weight tying*). Por isso
  ``wte.weight`` aparece duas vezes na tabela.

O GPT-2 usa bias nas projeções Q, K e V e contexto de 1024 tokens, então o
modelo que recebe os pesos precisa de ``qkv_bias=True`` e
``context_length=1024`` — diferente da configuração treinada na Sprint 5.

Os arquivos ``*.safetensors`` não são versionados no Git (ver ``.gitignore``).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

__all__ = [
    "PASTA_MODELOS",
    "REPOSITORIOS_GPT2",
    "caminho_dos_pesos",
    "baixar_pesos_gpt2",
    "Correspondencia",
    "mapa_de_parametros",
    "eh_tensor_ignorado",
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


# ---------------------------------------------------------------------------
# 2. Correspondência entre o arquivo do GPT-2 e o GPTModel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Correspondencia:
    """Uma linha da tabela arquivo -> modelo.

    Attributes:
        destino: nome do parâmetro no ``state_dict`` do ``GPTModel``.
        origem: nome do tensor no arquivo do GPT-2.
        transpor: ``True`` para pesos de ``Conv1D``, guardados como
            ``(entrada, saída)``.
        fatia: ``None`` para usar o tensor inteiro; 0, 1 ou 2 para ficar com o
            terço de *query*, *key* ou *value* de ``c_attn``.
    """

    destino: str
    origem: str
    transpor: bool = False
    fatia: int | None = None


def mapa_de_parametros(cfg: dict) -> list[Correspondencia]:
    """Tabela completa arquivo -> modelo para a configuração ``cfg``.

    Cada parâmetro do ``GPTModel`` aparece exatamente uma vez como destino.
    Os bias de Q, K e V só entram se ``cfg["qkv_bias"]`` for verdadeiro.
    """
    C = Correspondencia
    mapa = [
        C("embedding.token_emb.weight", "wte.weight"),
        C("embedding.pos_emb.weight", "wpe.weight"),
    ]
    for i in range(cfg["n_layers"]):
        bloco, h = f"trf_blocks.{i}", f"h.{i}"
        mapa += [
            C(f"{bloco}.norm1.scale", f"{h}.ln_1.weight"),
            C(f"{bloco}.norm1.shift", f"{h}.ln_1.bias"),
        ]
        for fatia, nome in enumerate(("W_query", "W_key", "W_value")):
            mapa.append(
                C(f"{bloco}.att.{nome}.weight", f"{h}.attn.c_attn.weight", True, fatia)
            )
            if cfg.get("qkv_bias", False):
                mapa.append(
                    C(f"{bloco}.att.{nome}.bias", f"{h}.attn.c_attn.bias", False, fatia)
                )
        mapa += [
            C(f"{bloco}.att.out_proj.weight", f"{h}.attn.c_proj.weight", True),
            C(f"{bloco}.att.out_proj.bias", f"{h}.attn.c_proj.bias"),
            C(f"{bloco}.norm2.scale", f"{h}.ln_2.weight"),
            C(f"{bloco}.norm2.shift", f"{h}.ln_2.bias"),
            C(f"{bloco}.ff.layers.0.weight", f"{h}.mlp.c_fc.weight", True),
            C(f"{bloco}.ff.layers.0.bias", f"{h}.mlp.c_fc.bias"),
            C(f"{bloco}.ff.layers.2.weight", f"{h}.mlp.c_proj.weight", True),
            C(f"{bloco}.ff.layers.2.bias", f"{h}.mlp.c_proj.bias"),
        ]
    mapa += [
        C("final_norm.scale", "ln_f.weight"),
        C("final_norm.shift", "ln_f.bias"),
        C("out_head.weight", "wte.weight"),
    ]
    return mapa


def eh_tensor_ignorado(nome: str) -> bool:
    """Tensores do arquivo que não são pesos treinados.

    ``h.i.attn.bias`` e ``h.i.attn.masked_bias`` são, apesar do nome, a máscara
    causal e a constante usada para mascarar — equivalentes ao buffer ``mask``
    da nossa ``MultiHeadAttention``, que é recriado na construção do modelo.
    ``lm_head.weight``, quando presente, é uma cópia de ``wte.weight``.
    """
    return nome.endswith((".attn.bias", ".attn.masked_bias")) or nome == "lm_head.weight"

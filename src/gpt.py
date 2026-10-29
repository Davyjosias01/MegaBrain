"""Arquitetura GPT — Sprint 4 (Capítulo 4 do livro-texto).

Este módulo integra os componentes das sprints anteriores em um modelo GPT
completo, seguindo o Capítulo 4 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*:

1. ``LayerNorm``        — normalização de cada vetor de token (seção 4.2);
2. ``GELU``             — função de ativação suave (seção 4.3);
3. ``FeedForward``      — rede de duas camadas aplicada a cada posição (seção 4.3);
4. ``TransformerBlock`` — atenção + feed forward, com LayerNorm e conexões
   residuais (seções 4.4 e 4.5);
5. ``GPTModel``         — embeddings, ``n_layers`` blocos empilhados, LayerNorm
   final e a cabeça de saída que produz os logits (seção 4.6).

Reaproveita ``EmbeddingDeEntrada`` (``src/embeddings.py``, Sprint 2) e
``MultiHeadAttention`` (``src/attention.py``, Sprint 3).

Caminho dos tensores no modelo::

    token IDs                      (b, L)              inteiros
      -> embedding de token + pos  (b, L, d)           Sprint 2
      -> dropout
      -> n_layers × TransformerBlock (b, L, d)         a forma não muda
      -> LayerNorm final           (b, L, d)
      -> cabeça de saída (linear)  (b, L, V)           logits

``b`` = lote, ``L`` = número de tokens, ``d`` = ``emb_dim``, ``V`` = ``vocab_size``.
Para cada uma das ``L`` posições o modelo produz ``V`` logits: uma pontuação
para cada token do vocabulário ser o *próximo* depois daquela posição.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from embeddings import EmbeddingDeEntrada

__all__ = [
    "CONFIG_GPT2_124M",
    "TAMANHOS_GPT2",
    "config_gpt2",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "RedeComAtalhos",
    "contar_parametros",
    "parametros_por_componente",
    "estimar_parametros",
    "tamanho_em_mb",
]

# Configuração do GPT-2 "small" (124M), usada ao longo do Capítulo 4.
CONFIG_GPT2_124M = {
    "vocab_size": 50257,     # tokens do BPE do GPT-2
    "context_length": 1024,  # maior sequência que o modelo aceita
    "emb_dim": 768,          # dimensão dos embeddings e de todos os vetores internos
    "n_heads": 12,           # cabeças de atenção por bloco
    "n_layers": 12,          # blocos Transformer empilhados
    "drop_rate": 0.1,        # taxa de dropout
    "qkv_bias": False,       # bias nas projeções Q, K e V
}

# (emb_dim, n_layers, n_heads) dos quatro tamanhos do GPT-2 (exercício 4.2).
TAMANHOS_GPT2 = {
    "small": (768, 12, 12),
    "medium": (1024, 24, 16),
    "large": (1280, 36, 20),
    "xl": (1600, 48, 25),
}


def config_gpt2(tamanho: str = "small", **alteracoes) -> dict:
    """Configuração de um dos tamanhos do GPT-2, com alterações opcionais.

    >>> config_gpt2("medium")["n_layers"]
    24
    >>> config_gpt2("small", n_layers=4)["n_layers"]
    4
    """
    emb_dim, n_layers, n_heads = TAMANHOS_GPT2[tamanho]
    cfg = dict(CONFIG_GPT2_124M, emb_dim=emb_dim, n_layers=n_layers, n_heads=n_heads)
    cfg.update(alteracoes)
    return cfg


# ---------------------------------------------------------------------------
# 1. Layer normalization (seção 4.2)
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Normaliza cada vetor de token para média 0 e variância 1 — listagem 4.2.

    A normalização é feita **na última dimensão**, isto é, sobre as ``emb_dim``
    componentes de cada token, de forma independente para cada token e cada
    exemplo do lote. Ao contrário da *batch normalization*, não depende dos
    outros exemplos do lote — o que a torna adequada a lotes pequenos e à
    geração, em que o lote tem um único exemplo.

    Depois de normalizar, dois parâmetros treináveis, ``scale`` (iniciado em 1)
    e ``shift`` (iniciado em 0), permitem ao modelo reescalar e deslocar o
    resultado se isso ajudar. Na inicialização a camada é uma normalização pura.

    ``eps`` evita divisão por zero quando a variância é nula. A variância é a
    populacional (``unbiased=False``), como no GPT-2 original.

    Formas: entrada e saída ``(..., emb_dim)``.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        media = x.mean(dim=-1, keepdim=True)
        variancia = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalizado = (x - media) / torch.sqrt(variancia + self.eps)
        return self.scale * x_normalizado + self.shift


# ---------------------------------------------------------------------------
# 2. GELU e feed forward (seção 4.3)
# ---------------------------------------------------------------------------
class GELU(nn.Module):
    """Gaussian Error Linear Unit, na aproximação por tangente hiperbólica — listagem 4.3.

    ``GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))``

    Parecida com a ReLU para ``|x|`` grande, mas suave em torno de zero e com
    saída levemente negativa para entradas negativas pequenas. A ReLU zera
    tudo o que é negativo e tem derivada nula ali: neurônios que caem nessa
    região deixam de aprender. A GELU mantém um gradiente pequeno, mas não nulo.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """Rede de duas camadas lineares com GELU no meio — listagem 4.4.

    Aplicada a cada token separadamente (os tokens não se misturam aqui; isso é
    papel da atenção). Expande o vetor para ``ff_dim`` dimensões, aplica a não
    linearidade e comprime de volta::

        (b, L, d) -> Linear d->ff_dim -> (b, L, ff_dim) -> GELU -> Linear ff_dim->d -> (b, L, d)

    A expansão dá à rede um espaço mais rico para combinar as características
    que a atenção reuniu. No GPT-2, ``ff_dim = 4 · emb_dim`` (768 -> 3.072), o
    padrão adotado aqui quando a configuração não define ``ff_dim``.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        emb_dim = cfg["emb_dim"]
        ff_dim = cfg.get("ff_dim", 4 * emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            GELU(),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# 3. Bloco Transformer (seções 4.4 e 4.5)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """O bloco que se repete ``n_layers`` vezes no GPT — listagem 4.6.

    Duas subcamadas, cada uma no padrão *pre-LayerNorm* com conexão residual::

        x -> LayerNorm -> MultiHeadAttention -> Dropout -> (+ x)   # mistura os tokens
          -> LayerNorm -> FeedForward        -> Dropout -> (+ x)   # processa cada token

    | Etapa                     | Forma           | Função                                         |
    |---------------------------|-----------------|------------------------------------------------|
    | entrada ``x``             | ``(b, L, d)``   | um vetor por token                             |
    | ``norm1``                 | ``(b, L, d)``   | estabiliza a escala antes da atenção            |
    | ``att`` (máscara causal)  | ``(b, L, d)``   | cada token reúne informação do seu passado     |
    | pesos de atenção internos | ``(b, h, L, L)``| ``h = n_heads``; um mapa por cabeça             |
    | ``drop_shortcut`` + ``x`` | ``(b, L, d)``   | conexão residual: soma a entrada de volta      |
    | ``norm2``                 | ``(b, L, d)``   | estabiliza a escala antes da feed forward       |
    | ``ff`` (expansão interna) | ``(b, L, 4d)``  | transformação não linear de cada token         |
    | ``drop_shortcut`` + atalho| ``(b, L, d)``   | segunda conexão residual                       |

    A forma de saída é igual à de entrada: é isso que permite empilhar blocos.

    **Conexões residuais.** Somar a entrada à saída de cada subcamada cria um
    caminho direto para o gradiente, que de outra forma seria multiplicado por
    dezenas de matrizes no caminho de volta e tenderia a desaparecer. Também
    faz cada bloco aprender uma *correção* sobre a representação que recebe, em
    vez de reconstruí-la do zero.

    **Flags experimentais.** ``usar_residual`` e ``usar_layernorm`` (lidas da
    configuração, padrão ``True``) desligam cada mecanismo, para os
    experimentos que medem sua contribuição para a estabilidade.

    A ordem de criação dos submódulos é a do livro, pois define o consumo do
    gerador aleatório e, portanto, a reprodutibilidade a partir de uma semente.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.usar_residual = cfg.get("usar_residual", True)
        usar_layernorm = cfg.get("usar_layernorm", True)

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"]) if usar_layernorm else nn.Identity()
        self.norm2 = LayerNorm(cfg["emb_dim"]) if usar_layernorm else nn.Identity()
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Subcamada 1: atenção causal multi-cabeça.
        atalho = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        if self.usar_residual:
            x = x + atalho

        # Subcamada 2: feed forward, aplicada a cada token.
        atalho = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        if self.usar_residual:
            x = x + atalho
        return x


# ---------------------------------------------------------------------------
# 4. Modelo GPT (seção 4.6)
# ---------------------------------------------------------------------------
class GPTModel(nn.Module):
    """Modelo GPT completo — listagem 4.7.

    Componentes, na ordem em que a sequência os atravessa:

    1. ``embedding``  — ``EmbeddingDeEntrada`` da Sprint 2: tabela de tokens
       ``(V, d)`` + tabela posicional ``(context_length, d)``;
    2. ``drop_emb``   — dropout sobre a soma dos embeddings;
    3. ``trf_blocks`` — ``n_layers`` blocos ``TransformerBlock`` em sequência;
    4. ``final_norm`` — LayerNorm aplicada à saída do último bloco;
    5. ``out_head``   — a *language model head*: ``Linear(d, V)`` sem bias, que
       projeta cada vetor de volta para o tamanho do vocabulário.

    ``forward`` recebe token IDs ``(b, L)`` e devolve logits ``(b, L, V)``.
    Os logits ainda não são probabilidades: o softmax é aplicado só na geração
    (``src/geracao.py``) e, no treinamento da Sprint 5, dentro da função de perda.

    **Compartilhamento de pesos (weight tying).** A cabeça de saída tem a
    mesma forma ``(V, d)`` da tabela de tokens. O GPT-2 original usa uma única
    matriz para as duas, o que economiza 38,6 milhões de parâmetros no tamanho
    small. O livro mantém as duas separadas, que é o padrão aqui; a chave
    ``compartilhar_pesos=True`` na configuração ativa o compartilhamento.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.embedding = EmbeddingDeEntrada(
            cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"]
        )
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = (
            LayerNorm(cfg["emb_dim"]) if cfg.get("usar_layernorm", True) else nn.Identity()
        )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("compartilhar_pesos", False):
            self.out_head.weight = self.embedding.token_emb.weight

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """``(b, L)`` token IDs -> ``(b, L, vocab_size)`` logits."""
        x = self.embedding(in_idx)   # (b, L, d)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)       # (b, L, d)
        x = self.final_norm(x)
        return self.out_head(x)      # (b, L, V)


# ---------------------------------------------------------------------------
# 5. Rede didática para as conexões residuais (seção 4.4)
# ---------------------------------------------------------------------------
class RedeComAtalhos(nn.Module):
    """Rede de 5 camadas ``Linear + GELU``, com atalhos opcionais — listagem 4.5.

    Não faz parte do GPT: isola o efeito das conexões residuais sobre o
    gradiente em uma rede pequena, em que ele pode ser inspecionado camada a
    camada. O atalho só é aplicado quando entrada e saída da camada têm a mesma
    forma.
    """

    def __init__(self, tamanhos: list[int], usar_atalhos: bool) -> None:
        super().__init__()
        self.usar_atalhos = usar_atalhos
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(tamanhos[i], tamanhos[i + 1]), GELU())
            for i in range(len(tamanhos) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for camada in self.layers:
            saida = camada(x)
            x = x + saida if self.usar_atalhos and x.shape == saida.shape else saida
        return x


# ---------------------------------------------------------------------------
# 6. Contagem de parâmetros e memória (seção 4.6)
# ---------------------------------------------------------------------------
def contar_parametros(modelo: nn.Module, sem_cabeca: bool = False) -> int:
    """Número de parâmetros distintos do modelo.

    Parâmetros compartilhados são contados uma vez. ``sem_cabeca=True`` desconta
    a cabeça de saída, o que equivale à contagem com compartilhamento de pesos
    usada para chamar o modelo de "124M".
    """
    total = sum(p.numel() for p in modelo.parameters())
    if sem_cabeca and isinstance(modelo, GPTModel):
        if modelo.out_head.weight is not modelo.embedding.token_emb.weight:
            total -= modelo.out_head.weight.numel()
    return total


def parametros_por_componente(modelo: GPTModel) -> dict[str, int]:
    """Distribuição dos parâmetros entre os componentes da arquitetura."""
    blocos = list(modelo.trf_blocks)

    def soma(modulos) -> int:
        return sum(p.numel() for m in modulos for p in m.parameters())

    partes = {
        "embedding de token": modelo.embedding.token_emb.weight.numel(),
        "embedding posicional": modelo.embedding.pos_emb.weight.numel(),
        "atenção (todos os blocos)": soma(b.att for b in blocos),
        "feed forward (todos os blocos)": soma(b.ff for b in blocos),
        "LayerNorm (blocos + final)": soma([b.norm1 for b in blocos] + [b.norm2 for b in blocos])
        + soma([modelo.final_norm]),
        "cabeça de saída": 0
        if modelo.out_head.weight is modelo.embedding.token_emb.weight
        else modelo.out_head.weight.numel(),
    }
    assert sum(partes.values()) == contar_parametros(modelo)
    return partes


def estimar_parametros(cfg: dict) -> int:
    """Parâmetros de um ``GPTModel`` calculados pela configuração, sem instanciá-lo.

    Permite conferir modelos grandes demais para a memória disponível (o
    GPT-2 XL ocupa mais de 6 GB em float32). Cada bloco tem:

    - atenção: 3 projeções ``d × d`` (+ bias opcional) e ``out_proj`` ``d × d + d``;
    - feed forward: ``d × f + f`` e ``f × d + d``;
    - duas LayerNorm com ``2d`` parâmetros cada.
    """
    v, c, d = cfg["vocab_size"], cfg["context_length"], cfg["emb_dim"]
    f = cfg.get("ff_dim", 4 * d)
    atencao = 3 * (d * d + (d if cfg["qkv_bias"] else 0)) + d * d + d
    feed_forward = d * f + f + f * d + d
    layernorm = 4 * d if cfg.get("usar_layernorm", True) else 0
    bloco = atencao + feed_forward + layernorm
    final = 2 * d if cfg.get("usar_layernorm", True) else 0
    cabeca = 0 if cfg.get("compartilhar_pesos", False) else v * d
    return v * d + c * d + cfg["n_layers"] * bloco + final + cabeca


def tamanho_em_mb(num_parametros: int, bytes_por_parametro: int = 4) -> float:
    """Memória dos pesos, em MB (float32 = 4 bytes por parâmetro)."""
    return num_parametros * bytes_por_parametro / (1024 * 1024)

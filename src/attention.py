"""Mecanismos de atenção — Sprint 3 (Capítulo 3 do livro-texto).

Este módulo implementa, de forma incremental, os quatro mecanismos de atenção
estudados no Capítulo 3 de RASCHKA, S. *Build a Large Language Model (From
Scratch)*:

1. ``atencao_simples``              — self-attention sem pesos treináveis (seção 3.3);
2. ``scaled_dot_product_attention`` — núcleo matemático da atenção (seção 3.4);
   ``SelfAttentionV1`` / ``SelfAttentionV2`` — self-attention com pesos treináveis;
3. ``CausalAttention``              — atenção mascarada, com dropout (seção 3.5);
4. ``MultiHeadAttentionWrapper`` e ``MultiHeadAttention`` — múltiplas cabeças (seção 3.6).

Todas as classes derivam de ``torch.nn.Module`` e podem ser plugadas nos blocos
Transformer da Sprint 4.

Convenções usadas em todo o módulo
----------------------------------
``b``            tamanho do lote (batch);
``num_tokens``   número de tokens da sequência;
``d_in``         dimensão do embedding de entrada;
``d_out``        dimensão do vetor de contexto produzido;
``num_heads``    número de cabeças de atenção;
``head_dim``     ``d_out // num_heads``.

Os tensores de entrada podem ser 2D ``(num_tokens, d_in)`` — uma única sequência —
ou 3D ``(b, num_tokens, d_in)`` — um lote vindo do DataLoader da Sprint 2.

Todos os ``forward`` aceitam ``return_attn_weights``. Quando ``True``, devolvem
também a matriz de pesos de atenção, o que permite visualizá-la e inspecioná-la
nos experimentos sem alterar o contrato padrão (que devolve apenas os vetores de
contexto, como esperado pelos blocos Transformer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "atencao_simples",
    "mascara_causal",
    "scaled_dot_product_attention",
    "SelfAttentionV1",
    "SelfAttentionV2",
    "CausalAttention",
    "MultiHeadAttentionWrapper",
    "MultiHeadAttention",
]


# ---------------------------------------------------------------------------
# 1. Self-attention sem pesos treináveis (seção 3.3)
# ---------------------------------------------------------------------------
def atencao_simples(entradas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-attention simplificada, sem nenhum parâmetro treinável.

    É a versão didática da seção 3.3: o próprio embedding do token faz,
    simultaneamente, o papel de query, key e value. Os três passos são:

    1. **scores** — produto escalar de cada token com todos os outros.
       O produto escalar mede alinhamento entre vetores: quanto maior, mais
       parecidos os dois embeddings e maior a atenção entre eles.
    2. **pesos**  — softmax por linha, para que cada linha some 1.
    3. **contexto** — soma ponderada dos próprios embeddings de entrada.

    Como não há matrizes ``W``, nada aqui é aprendido: a relação entre os tokens
    depende exclusivamente da geometria dos embeddings. É justamente essa
    limitação que motiva a introdução dos pesos treináveis na seção 3.4.

    Args:
        entradas: tensor ``(num_tokens, d_in)`` ou ``(b, num_tokens, d_in)``.

    Returns:
        ``(vetores_de_contexto, pesos_de_atencao)``, com formas
        ``(..., num_tokens, d_in)`` e ``(..., num_tokens, num_tokens)``.
    """
    if entradas.dim() not in (2, 3):
        raise ValueError(f"entradas deve ser 2D ou 3D, recebido {entradas.dim()}D.")

    attn_scores = entradas @ entradas.transpose(-2, -1)      # omega
    attn_weights = torch.softmax(attn_scores, dim=-1)        # alpha
    context_vec = attn_weights @ entradas                    # z
    return context_vec, attn_weights


# ---------------------------------------------------------------------------
# 2. Scaled dot-product attention (seção 3.4)
# ---------------------------------------------------------------------------
def mascara_causal(
    num_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Devolve a máscara causal booleana ``(num_tokens, num_tokens)``.

    ``True`` marca as posições que devem ser escondidas — o triângulo *acima* da
    diagonal, isto é, os tokens futuros (seção 3.5).
    """
    return torch.triu(
        torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=device),
        diagonal=1,
    )


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
    scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Núcleo compartilhado por todos os mecanismos com pesos treináveis.

    Implementa ``softmax(Q @ K.T / sqrt(d_k) + mascara) @ V``, ou seja::

        attn_scores  = Q @ K.T                    (similaridade query x key)
        attn_scores /= sqrt(d_k)                  (escala — seção 3.4.1)
        attn_scores  = masked_fill(-inf)          (máscara causal — seção 3.5)
        attn_weights = softmax(attn_scores)       (distribuição que soma 1)
        contexto     = attn_weights @ V           (soma ponderada dos values)

    **Por que dividir por sqrt(d_k).** O produto escalar de dois vetores de
    dimensão ``d_k`` cresce, em média, proporcionalmente a ``sqrt(d_k)``. Scores
    muito grandes empurram o softmax para um comportamento de função degrau: um
    peso próximo de 1 e os demais próximos de 0. Nessa região a derivada do
    softmax é quase nula, os gradientes somem e o treinamento estagna. Dividir
    por ``sqrt(d_k)`` mantém a variância dos scores aproximadamente constante e,
    com isso, a distribuição de atenção utilizável para o aprendizado. É essa
    divisão que dá nome ao mecanismo: *scaled* dot-product attention.

    **Por que -inf na máscara.** ``exp(-inf) = 0``, portanto o softmax atribui
    peso exatamente zero às posições mascaradas e já renormaliza as restantes,
    dispensando a renormalização manual descrita na figura 3.20.

    Args:
        queries: ``(..., num_tokens_q, d_k)``.
        keys:    ``(..., num_tokens_k, d_k)``.
        values:  ``(..., num_tokens_k, d_v)``.
        mask:    tensor booleano difundível para ``(..., num_tokens_q,
                 num_tokens_k)``; ``True`` esconde a posição.
        dropout: camada ``nn.Dropout`` aplicada *depois* do softmax.
        scale:   se ``False``, omite a divisão por ``sqrt(d_k)`` (usado no
                 experimento que compara atenção com e sem escala).

    Returns:
        ``(vetores_de_contexto, pesos_de_atencao)``.
    """
    d_k = keys.shape[-1]

    attn_scores = queries @ keys.transpose(-2, -1)
    if scale:
        attn_scores = attn_scores / d_k**0.5
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

    attn_weights = torch.softmax(attn_scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    context_vec = attn_weights @ values
    return context_vec, attn_weights


class SelfAttentionV1(nn.Module):
    """Self-attention com pesos treináveis declarados como ``nn.Parameter``.

    Corresponde à listagem 3.1. As três matrizes ``W_query``, ``W_key`` e
    ``W_value`` projetam cada embedding de entrada em três papéis distintos:

    - **query** — o que o token *procura* nos demais;
    - **key**   — o que o token *oferece* para ser encontrado;
    - **value** — o conteúdo que será efetivamente somado no vetor de contexto.

    Separar esses papéis é o que permite ao modelo aprender relações assimétricas
    (o token *A* pode atender fortemente *B* sem que a recíproca seja verdadeira),
    algo impossível em ``atencao_simples``, onde a matriz de scores é simétrica.

    Os pesos são inicializados com ``torch.rand`` (distribuição uniforme em
    ``[0, 1)``), exatamente como no livro. Essa inicialização é propositalmente
    ingênua — ``SelfAttentionV2`` a substitui pela do ``nn.Linear``.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # A ordem de criação define o consumo do gerador aleatório; mantida
        # igual à do livro para que os resultados sejam reproduzíveis.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        context_vec, attn_weights = scaled_dot_product_attention(queries, keys, values)
        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec


class SelfAttentionV2(nn.Module):
    """Self-attention com pesos treináveis declarados como ``nn.Linear``.

    Corresponde à listagem 3.2. Matematicamente é idêntica a ``SelfAttentionV1``
    — uma camada linear sem bias *é* uma multiplicação de matrizes —, mas o
    ``nn.Linear`` usa a inicialização de Kaiming uniforme, que mantém a variância
    das ativações sob controle e torna o treinamento mais estável. Por isso as
    duas classes produzem números diferentes a partir da mesma semente.

    Atenção a um detalhe de implementação: ``nn.Linear`` guarda a matriz de pesos
    **transposta**, com forma ``(d_out, d_in)``. Logo, ``W_query.weight.T``
    equivale ao ``W_query`` de ``SelfAttentionV1`` (exercício 3.1 do livro).

    Esta é a classe que o livro chama de *scaled dot-product attention*: a
    self-attention completa usada no Transformer original e nos modelos GPT,
    ainda sem máscara causal.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
        scale: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``scale=False`` desliga a divisão por ``sqrt(d_k)`` (experimento 06)."""
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        context_vec, attn_weights = scaled_dot_product_attention(
            queries, keys, values, scale=scale
        )
        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec


# ---------------------------------------------------------------------------
# 3. Causal attention (seção 3.5)
# ---------------------------------------------------------------------------
class CausalAttention(nn.Module):
    """Self-attention mascarada, com dropout — listagem 3.3.

    Acrescenta duas coisas a ``SelfAttentionV2``:

    **Máscara causal.** Um LLM autorregressivo é treinado para prever o próximo
    token. Se, ao calcular o vetor de contexto da posição *i*, o modelo pudesse
    olhar as posições *i+1, i+2, …*, ele estaria lendo a resposta: aprenderia a
    copiar o futuro em vez de prevê-lo, e falharia na inferência, quando o futuro
    não existe. A máscara zera os pesos acima da diagonal, restringindo cada
    token ao seu próprio passado e a si mesmo.

    **Dropout.** Aplicado *sobre os pesos de atenção*, já normalizados. Anula
    aleatoriamente uma fração ``p`` deles e multiplica os sobreviventes por
    ``1/(1-p)``, preservando a escala esperada da saída. Impede que o modelo
    fique dependente de um punhado fixo de ligações entre tokens. Só age em modo
    de treino: ``module.eval()`` o desativa.

    A máscara é registrada com ``register_buffer``: ela é estado do módulo, mas
    não é parâmetro treinável. Com isso acompanha o modelo em ``.to(device)`` e
    aparece no ``state_dict``, sem receber gradiente.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", mascara_causal(context_length))

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
        usar_mascara: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``usar_mascara=False`` desliga a máscara causal (experimento 07)."""
        num_tokens = x.shape[-2]
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequência com {num_tokens} tokens excede o context_length "
                f"de {self.context_length}."
            )

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # A máscara é recortada para o tamanho real da sequência, de modo que o
        # módulo aceite lotes mais curtos que o context_length.
        mask = self.mask[:num_tokens, :num_tokens] if usar_mascara else None

        context_vec, attn_weights = scaled_dot_product_attention(
            queries, keys, values, mask=mask, dropout=self.dropout
        )
        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec


# ---------------------------------------------------------------------------
# 4. Multi-head attention (seção 3.6)
# ---------------------------------------------------------------------------
class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention por empilhamento — listagem 3.4.

    Versão didática (seção 3.6.1): instancia ``num_heads`` módulos
    ``CausalAttention`` independentes, cada um com seu próprio trio de matrizes
    ``W``, e concatena as saídas na última dimensão. Cada cabeça enxerga a mesma
    sequência sob uma projeção diferente e pode, portanto, especializar-se em um
    tipo de relação (proximidade, concordância, pontuação, …).

    A dimensão final é ``num_heads * d_out``, e as cabeças são executadas em
    laço Python, uma após a outra. As duas características são corrigidas na
    classe ``MultiHeadAttention``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_out
        self.d_out = d_out * num_heads
        self.heads = nn.ModuleList(
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        )

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        saidas = [h(x, return_attn_weights=True) for h in self.heads]
        context_vec = torch.cat([s[0] for s in saidas], dim=-1)
        if return_attn_weights:
            # (..., num_heads, num_tokens, num_tokens) — mesma organização da
            # MultiHeadAttention, para que as visualizações sejam comparáveis.
            attn_weights = torch.stack([s[1] for s in saidas], dim=-3)
            return context_vec, attn_weights
        return context_vec


class MultiHeadAttention(nn.Module):
    """Multi-head attention eficiente, por divisão de pesos — listagem 3.5.

    Em vez de manter ``num_heads`` módulos separados, usa **uma** projeção linear
    de tamanho ``d_out`` para queries, keys e values e depois fatia o resultado
    em ``num_heads`` blocos de ``head_dim = d_out // num_heads`` colunas. Isso é
    matematicamente equivalente ao empilhamento, porque cada bloco de colunas de
    ``W_query`` é independente dos demais — mas troca ``3 * num_heads``
    multiplicações de matriz por apenas 3, o passo mais caro do mecanismo.

    O caminho dos tensores no ``forward`` é::

        (b, num_tokens, d_in)
          -> W_q/W_k/W_v            -> (b, num_tokens, d_out)
          -> view                   -> (b, num_tokens, num_heads, head_dim)
          -> transpose(1, 2)        -> (b, num_heads, num_tokens, head_dim)
          -> atenção por cabeça     -> (b, num_heads, num_tokens, head_dim)
          -> transpose(1, 2)        -> (b, num_tokens, num_heads, head_dim)
          -> view                   -> (b, num_tokens, d_out)
          -> out_proj               -> (b, num_tokens, d_out)

    O ``transpose(1, 2)`` coloca ``num_heads`` antes de ``num_tokens``: o ``@``
    do PyTorch trata as dimensões da frente como dimensões de lote e aplica a
    multiplicação às duas últimas, executando todas as cabeças de todos os
    exemplos em uma única chamada.

    ``out_proj`` é a projeção de saída que mistura as informações das cabeças
    antes de devolvê-las. Não é estritamente necessária, mas é padrão nas
    arquiteturas de LLM e está presente no GPT.

    Diferente de ``MultiHeadAttentionWrapper``, aqui ``d_out`` é a dimensão
    **total** da saída, e não a de cada cabeça.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) deve ser divisível por num_heads ({num_heads})."
            )

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", mascara_causal(context_length))

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
        usar_mascara: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(
                "MultiHeadAttention espera um tensor 3D "
                f"(b, num_tokens, d_in); recebido {x.dim()}D."
            )

        b, num_tokens, _ = x.shape
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequência com {num_tokens} tokens excede o context_length "
                f"de {self.context_length}."
            )

        # Uma única projeção por papel, depois fatiada entre as cabeças.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        def dividir_em_cabecas(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        queries = dividir_em_cabecas(queries)
        keys = dividir_em_cabecas(keys)
        values = dividir_em_cabecas(values)

        mask = self.mask[:num_tokens, :num_tokens] if usar_mascara else None

        # attn_weights: (b, num_heads, num_tokens, num_tokens)
        context_vec, attn_weights = scaled_dot_product_attention(
            queries, keys, values, mask=mask, dropout=self.dropout
        )

        # Reagrupa as cabeças lado a lado e mistura com a projeção de saída.
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec

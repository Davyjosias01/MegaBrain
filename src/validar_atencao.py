"""Validação numérica dos mecanismos de atenção da Sprint 3.

Confere a implementação de ``src/attention.py`` contra os valores de referência
publicados no Capítulo 3 do livro-texto e contra as propriedades matemáticas que
os mecanismos devem satisfazer.

Todos os testes usam o mesmo exemplo do livro — a frase *"Your journey starts
with one step"* já convertida em seis embeddings de 3 dimensões — e as mesmas
sementes (``123`` e ``789``), de modo que os números possam ser comparados
diretamente com as páginas indicadas em cada teste.

Execução, a partir da raiz do repositório::

    python src/validar_atencao.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import (  # noqa: E402
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttentionV1,
    SelfAttentionV2,
    atencao_simples,
    mascara_causal,
    scaled_dot_product_attention,
)

TOLERANCIA = 1e-4

# Seção 3.3.1, página 71: "Your journey starts with one step" em 3 dimensões.
ENTRADAS = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your    x^1
        [0.55, 0.87, 0.66],  # journey x^2
        [0.57, 0.85, 0.64],  # starts  x^3
        [0.22, 0.58, 0.33],  # with    x^4
        [0.77, 0.25, 0.10],  # one     x^5
        [0.05, 0.80, 0.55],  # step    x^6
    ]
)
LOTE = torch.stack((ENTRADAS, ENTRADAS), dim=0)  # (2, 6, 3), página 98

_resultados: list[tuple[bool, str]] = []


def _verificar(condicao: bool, descricao: str) -> None:
    _resultados.append((bool(condicao), descricao))
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}")


def _proximo(obtido: torch.Tensor, esperado: torch.Tensor) -> bool:
    return torch.allclose(obtido, esperado, atol=TOLERANCIA)


# ---------------------------------------------------------------------------
def teste_atencao_simples() -> None:
    """Seção 3.3 — self-attention sem pesos treináveis."""
    print("\n1. atencao_simples (seção 3.3, páginas 72-78)")

    contexto, pesos = atencao_simples(ENTRADAS)

    # Página 72: scores do token 2 usado como query.
    esperado_scores = torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
    scores = ENTRADAS @ ENTRADAS[1]
    _verificar(_proximo(scores, esperado_scores), "scores de x^2 conferem com a p. 72")

    # Página 74: pesos de atenção do token 2 após softmax.
    esperado_pesos = torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    _verificar(_proximo(pesos[1], esperado_pesos), "pesos de x^2 conferem com a p. 74")

    # Página 78: todos os vetores de contexto.
    esperado_contexto = torch.tensor(
        [
            [0.4421, 0.5931, 0.5790],
            [0.4419, 0.6515, 0.5683],
            [0.4431, 0.6496, 0.5671],
            [0.4304, 0.6298, 0.5510],
            [0.4671, 0.5910, 0.5266],
            [0.4177, 0.6503, 0.5645],
        ]
    )
    _verificar(
        _proximo(contexto, esperado_contexto),
        "vetores de contexto conferem com a p. 78",
    )

    _verificar(
        _proximo(pesos.sum(dim=-1), torch.ones(6)),
        "cada linha dos pesos soma 1",
    )
    scores_completos = ENTRADAS @ ENTRADAS.T
    _verificar(
        _proximo(scores_completos, scores_completos.T),
        "a matriz de scores é simétrica (não há W_q e W_k distintos)",
    )


def teste_self_attention_v1() -> None:
    """Listagem 3.1 — self-attention com nn.Parameter."""
    print("\n2. SelfAttentionV1 (listagem 3.1, página 87)")

    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(d_in=3, d_out=2)
    saida = sa_v1(ENTRADAS)

    esperado = torch.tensor(
        [
            [0.2996, 0.8053],
            [0.3061, 0.8210],
            [0.3058, 0.8203],
            [0.2948, 0.7939],
            [0.2927, 0.7891],
            [0.2990, 0.8040],
        ]
    )
    _verificar(_proximo(saida, esperado), "saída confere com a p. 87 (seed 123)")

    # Página 85: o vetor de contexto z^2 calculado passo a passo.
    _verificar(
        _proximo(saida[1], torch.tensor([0.3061, 0.8210])),
        "segunda linha reproduz o z^2 da p. 85",
    )


def teste_self_attention_v2() -> None:
    """Listagem 3.2 — self-attention com nn.Linear."""
    print("\n3. SelfAttentionV2 (listagem 3.2, página 89)")

    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in=3, d_out=2)
    saida = sa_v2(ENTRADAS)

    esperado = torch.tensor(
        [
            [-0.0739, 0.0713],
            [-0.0748, 0.0703],
            [-0.0749, 0.0702],
            [-0.0760, 0.0685],
            [-0.0763, 0.0679],
            [-0.0754, 0.0693],
        ]
    )
    _verificar(_proximo(saida, esperado), "saída confere com a p. 89 (seed 789)")

    # Exercício 3.1: transferir os pesos de V2 para V1 deve igualar as saídas.
    # nn.Linear guarda a matriz transposta, daí o .T.
    sa_v1 = SelfAttentionV1(d_in=3, d_out=2)
    with torch.no_grad():
        sa_v1.W_query.copy_(sa_v2.W_query.weight.T)
        sa_v1.W_key.copy_(sa_v2.W_key.weight.T)
        sa_v1.W_value.copy_(sa_v2.W_value.weight.T)
    _verificar(
        _proximo(sa_v1(ENTRADAS), saida),
        "exercício 3.1: V1 com os pesos de V2 produz a mesma saída",
    )


def teste_escala() -> None:
    """Seção 3.4.1 — o efeito da divisão por sqrt(d_k)."""
    print("\n4. scaled_dot_product_attention (seção 3.4.1, página 84)")

    torch.manual_seed(123)
    q = torch.randn(1, 512)
    k = torch.randn(64, 512)
    v = torch.randn(64, 8)

    _, pesos_com = scaled_dot_product_attention(q, k, v, scale=True)
    _, pesos_sem = scaled_dot_product_attention(q, k, v, scale=False)

    _verificar(
        pesos_com.max().item() < pesos_sem.max().item(),
        "sem escala o softmax concentra mais a massa em um único token",
    )

    def entropia(p: torch.Tensor) -> float:
        return float(-(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean())

    _verificar(
        entropia(pesos_com) > entropia(pesos_sem),
        "com escala a distribuição de atenção é mais entrópica",
    )

    # Equivalência: sem máscara, o núcleo reproduz a SelfAttentionV2.
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in=3, d_out=2)
    contexto_ref = sa_v2(ENTRADAS)
    contexto, _ = scaled_dot_product_attention(
        sa_v2.W_query(ENTRADAS), sa_v2.W_key(ENTRADAS), sa_v2.W_value(ENTRADAS)
    )
    _verificar(
        _proximo(contexto, contexto_ref),
        "o núcleo reproduz a SelfAttentionV2 quando não há máscara",
    )


def teste_causal_attention() -> None:
    """Listagem 3.3 — atenção causal."""
    print("\n5. CausalAttention (listagem 3.3, páginas 92-99)")

    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    contexto, pesos = ca(LOTE, return_attn_weights=True)

    _verificar(
        tuple(contexto.shape) == (2, 6, 2),
        "formato do vetor de contexto é (2, 6, 2), como na p. 99",
    )

    triangulo_superior = pesos[0].triu(diagonal=1)
    _verificar(
        bool((triangulo_superior == 0).all()),
        "todos os pesos acima da diagonal são exatamente zero",
    )
    _verificar(
        _proximo(pesos[0].sum(dim=-1), torch.ones(6)),
        "cada linha ainda soma 1 após a máscara",
    )

    # Figura 3.20 (mascarar depois do softmax e renormalizar) vs. figura 3.21
    # (mascarar com -inf antes do softmax): resultados idênticos.
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in=3, d_out=2)
    scores = sa_v2.W_query(ENTRADAS) @ sa_v2.W_key(ENTRADAS).T / 2**0.5

    via_figura_320 = torch.softmax(scores, dim=-1) * torch.tril(torch.ones(6, 6))
    via_figura_320 = via_figura_320 / via_figura_320.sum(dim=-1, keepdim=True)
    via_figura_321 = torch.softmax(
        scores.masked_fill(mascara_causal(6), -torch.inf), dim=-1
    )
    _verificar(
        _proximo(via_figura_320, via_figura_321),
        "as duas estratégias de máscara (fig. 3.20 e 3.21) são equivalentes",
    )

    # Página 93: os pesos mascarados publicados no livro.
    esperado = torch.tensor(
        [
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
            [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
            [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
        ]
    )
    _verificar(_proximo(via_figura_321, esperado), "pesos mascarados conferem com a p. 93")

    # Ausência de vazamento de informação: alterar um token futuro não pode
    # mudar o vetor de contexto de nenhuma posição anterior.
    ca.eval()
    alterado = ENTRADAS.clone()
    alterado[4] = torch.tensor([9.0, -9.0, 9.0])
    original = ca(ENTRADAS.unsqueeze(0))[0]
    perturbado = ca(alterado.unsqueeze(0))[0]
    _verificar(
        _proximo(original[:4], perturbado[:4]),
        "alterar o token 5 não afeta os contextos dos tokens 1 a 4",
    )
    _verificar(
        not _proximo(original[4:], perturbado[4:]),
        "alterar o token 5 afeta os contextos dos tokens 5 e 6",
    )


def teste_multi_head() -> None:
    """Listagens 3.4 e 3.5 — multi-head attention."""
    print("\n6. Multi-head attention (listagens 3.4 e 3.5, páginas 102-109)")

    torch.manual_seed(123)
    mhw = MultiHeadAttentionWrapper(
        d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2
    )
    saida_wrapper = mhw(LOTE)

    esperado_wrapper = torch.tensor(
        [
            [-0.4519, 0.2216, 0.4772, 0.1063],
            [-0.5874, 0.0058, 0.5891, 0.3257],
            [-0.6300, -0.0632, 0.6202, 0.3860],
            [-0.5675, -0.0843, 0.5478, 0.3589],
            [-0.5526, -0.0981, 0.5321, 0.3428],
            [-0.5299, -0.1081, 0.5077, 0.3493],
        ]
    ).expand(2, 6, 4)
    _verificar(
        _proximo(saida_wrapper, esperado_wrapper),
        "MultiHeadAttentionWrapper confere com a p. 103 (seed 123)",
    )
    _verificar(
        tuple(saida_wrapper.shape) == (2, 6, 4),
        "a saída do wrapper tem dimensão d_out * num_heads = 4",
    )

    torch.manual_seed(123)
    mha = MultiHeadAttention(
        d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2
    )
    saida_mha = mha(LOTE)

    esperado_mha = torch.tensor(
        [
            [0.3190, 0.4858],
            [0.2943, 0.3897],
            [0.2856, 0.3593],
            [0.2693, 0.3873],
            [0.2639, 0.3928],
            [0.2575, 0.4028],
        ]
    ).expand(2, 6, 2)
    _verificar(
        _proximo(saida_mha, esperado_mha),
        "MultiHeadAttention confere com a p. 109 (seed 123)",
    )
    _verificar(
        tuple(saida_mha.shape) == (2, 6, 2),
        "na MultiHeadAttention, d_out já é a dimensão total da saída",
    )

    _, pesos = mha(LOTE, return_attn_weights=True)
    _verificar(
        tuple(pesos.shape) == (2, 2, 6, 6),
        "os pesos saem como (b, num_heads, num_tokens, num_tokens)",
    )
    _verificar(
        bool((pesos.triu(diagonal=1) == 0).all()),
        "todas as cabeças respeitam a máscara causal",
    )

    # Equivalência entre as duas formas de montar as cabeças: o empilhamento e a
    # divisão de pesos são a mesma operação. Basta copiar as matrizes das duas
    # CausalAttention do wrapper para os blocos correspondentes da MHA e anular
    # a projeção de saída, que só existe na segunda.
    torch.manual_seed(123)
    ref = MultiHeadAttentionWrapper(
        d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2
    )
    equiv = MultiHeadAttention(
        d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2
    )
    with torch.no_grad():
        for nome in ("W_query", "W_key", "W_value"):
            blocos = [getattr(h, nome).weight for h in ref.heads]
            getattr(equiv, nome).weight.copy_(torch.cat(blocos, dim=0))
        equiv.out_proj.weight.copy_(torch.eye(4))
        equiv.out_proj.bias.zero_()
    _verificar(
        _proximo(equiv(LOTE), ref(LOTE)),
        "empilhar cabeças e dividir pesos produzem o mesmo resultado",
    )

    # O número de parâmetros das projeções Q/K/V não depende de num_heads.
    def n_params_qkv(m: MultiHeadAttention) -> int:
        return sum(
            p.numel()
            for nome, p in m.named_parameters()
            if nome.startswith(("W_query", "W_key", "W_value"))
        )

    contagens = set()
    for h in (1, 2, 4, 8):
        torch.manual_seed(123)
        contagens.add(
            n_params_qkv(
                MultiHeadAttention(
                    d_in=16, d_out=16, context_length=6, dropout=0.0, num_heads=h
                )
            )
        )
    _verificar(
        len(contagens) == 1,
        "com d_out fixo, mudar num_heads não altera a contagem de parâmetros",
    )

    # Exercício 3.3: dimensões do menor modelo GPT-2.
    gpt2 = MultiHeadAttention(
        d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=12
    )
    total = sum(p.numel() for p in gpt2.parameters())
    _verificar(
        gpt2.head_dim == 64 and total == 4 * 768 * 768 + 768,
        f"exercício 3.3: GPT-2 small tem head_dim 64 e {total:,} parâmetros",
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Sprint 3 — validação dos mecanismos de atenção (Capítulo 3)")
    print("=" * 72)

    teste_atencao_simples()
    teste_self_attention_v1()
    teste_self_attention_v2()
    teste_escala()
    teste_causal_attention()
    teste_multi_head()

    falhas = [d for ok, d in _resultados if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_resultados) - len(falhas)}/{len(_resultados)} verificações aprovadas")
    for d in falhas:
        print(f"  FALHOU: {d}")
    print("=" * 72)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())

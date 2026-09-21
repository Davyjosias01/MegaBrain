"""Experimento 04 — Self-Attention comparada a Multi-Head Attention.

**O que compara:** os quatro mecanismos do Capítulo 3, com ``d_in = d_out = 96``,
128 tokens e lote de 8:

1. ``atencao_simples``            — sem pesos treináveis (seção 3.3);
2. ``SelfAttentionV2``            — scaled dot-product, uma cabeça, sem máscara;
3. ``CausalAttention``            — uma cabeça, com máscara;
4. ``MultiHeadAttentionWrapper``  — 4 cabeças empilhadas (listagem 3.4);
5. ``MultiHeadAttention``         — 4 cabeças por divisão de pesos (listagem 3.5).

**Perguntas:**

- quanto custa cada mecanismo, em parâmetros e em tempo?
- as duas implementações de multi-head são equivalentes em resultado, mas quanto
  a divisão de pesos economiza em relação ao empilhamento?
- várias cabeças são realmente diferentes entre si, ou apenas repetem a mesma
  distribuição de atenção?

A última pergunta é a que justifica o mecanismo: se todas as cabeças produzissem
a mesma matriz de atenção, multi-head seria só desperdício. A medida usada é a
distância L1 média entre pares de cabeças.

**Ressalva importante.** Nada aqui foi treinado. As diferenças entre cabeças
observadas são consequência da inicialização aleatória independente de cada
bloco de ``W_query``/``W_key``. O experimento mostra que o mecanismo *permite*
comportamentos distintos, não que ele já tenha aprendido algum.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    GRADE,
    SEQUENCIAL,
    TINTA_SECUNDARIA,
    cabecalho,
    contar_parametros,
    cronometrar,
    dissimilaridade_entre_cabecas,
    entropia_normalizada,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import (
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttentionV2,
    atencao_simples,
)
from entradas_atencao import lote_de_embeddings

D_MODELO = 96
NUM_HEADS = 4
NUM_TOKENS = 128
BATCH = 8


def executar() -> list[dict]:
    cabecalho("Experimento 04 — Self-Attention x Multi-Head Attention")
    print(
        f"  d_in = d_out = {D_MODELO}  num_heads={NUM_HEADS}  "
        f"num_tokens={NUM_TOKENS}  batch={BATCH}"
    )

    dados = lote_de_embeddings(
        d_in=D_MODELO, max_length=NUM_TOKENS, batch_size=BATCH
    )
    x = dados["embeddings"]

    linhas: list[dict] = []

    # 1. Atenção simplificada, sem parâmetros.
    _, pesos = atencao_simples(x)
    linhas.append(
        {
            "mecanismo": "atencao_simples",
            "cabecas": 1,
            "d_saida": D_MODELO,
            "parametros": 0,
            "causal": "nao",
            "tempo_ms": cronometrar(lambda: atencao_simples(x)),
            "entropia_normalizada": entropia_normalizada(pesos, causal=False),
            "dissimilaridade_entre_cabecas": 0.0,
        }
    )

    # 2. Self-attention com pesos treináveis, uma cabeça, sem máscara.
    semear()
    sa = SelfAttentionV2(D_MODELO, D_MODELO).eval()
    _, pesos = sa(x, return_attn_weights=True)
    linhas.append(
        {
            "mecanismo": "SelfAttentionV2",
            "cabecas": 1,
            "d_saida": D_MODELO,
            "parametros": contar_parametros(sa),
            "causal": "nao",
            "tempo_ms": cronometrar(lambda: sa(x)),
            "entropia_normalizada": entropia_normalizada(pesos, causal=False),
            "dissimilaridade_entre_cabecas": 0.0,
        }
    )

    # 3. Atenção causal, uma cabeça.
    semear()
    ca = CausalAttention(D_MODELO, D_MODELO, NUM_TOKENS, 0.0).eval()
    _, pesos = ca(x, return_attn_weights=True)
    linhas.append(
        {
            "mecanismo": "CausalAttention",
            "cabecas": 1,
            "d_saida": D_MODELO,
            "parametros": contar_parametros(ca),
            "causal": "sim",
            "tempo_ms": cronometrar(lambda: ca(x)),
            "entropia_normalizada": entropia_normalizada(pesos),
            "dissimilaridade_entre_cabecas": 0.0,
        }
    )

    # 4. Multi-head por empilhamento. d_out por cabeça = D_MODELO // NUM_HEADS,
    #    para que a dimensão total da saída também seja D_MODELO.
    semear()
    mhw = MultiHeadAttentionWrapper(
        D_MODELO, D_MODELO // NUM_HEADS, NUM_TOKENS, 0.0, NUM_HEADS
    ).eval()
    _, pesos_mhw = mhw(x, return_attn_weights=True)
    linhas.append(
        {
            "mecanismo": "MultiHeadAttentionWrapper",
            "cabecas": NUM_HEADS,
            "d_saida": mhw.d_out,
            "parametros": contar_parametros(mhw),
            "causal": "sim",
            "tempo_ms": cronometrar(lambda: mhw(x)),
            "entropia_normalizada": entropia_normalizada(pesos_mhw),
            "dissimilaridade_entre_cabecas": dissimilaridade_entre_cabecas(pesos_mhw),
        }
    )

    # 5. Multi-head por divisão de pesos.
    semear()
    mha = MultiHeadAttention(D_MODELO, D_MODELO, NUM_TOKENS, 0.0, NUM_HEADS).eval()
    _, pesos_mha = mha(x, return_attn_weights=True)
    linhas.append(
        {
            "mecanismo": "MultiHeadAttention",
            "cabecas": NUM_HEADS,
            "d_saida": mha.d_out,
            "parametros": contar_parametros(mha),
            "causal": "sim",
            "tempo_ms": cronometrar(lambda: mha(x)),
            "entropia_normalizada": entropia_normalizada(pesos_mha),
            "dissimilaridade_entre_cabecas": dissimilaridade_entre_cabecas(pesos_mha),
        }
    )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp04-self-vs-multihead")

    print(
        "\n  Diferença entre as cabeças da MultiHeadAttention: "
        f"{linhas[4]['dissimilaridade_entre_cabecas']:.4f} "
        "(0 significaria cabeças redundantes)."
    )
    print(
        "  Observação: a MultiHeadAttention tem "
        f"{linhas[4]['parametros'] - linhas[3]['parametros']:,} parâmetros a mais "
        "que o wrapper — é a projeção de saída out_proj, que só ela possui."
    )

    _graficos(linhas, pesos_mha)
    _comparar_em_escala()
    return linhas


ESCALAS = [
    # (d_modelo, num_heads, num_tokens) — da configuração didática do capítulo
    # até as dimensões do menor GPT-2 (d = 768, 12 cabeças, p. 109).
    (96, 4, 128),
    (192, 4, 128),
    (384, 8, 256),
    (768, 12, 256),
]


def _comparar_em_escala() -> list[dict]:
    """Empilhamento x divisão de pesos em dimensões crescentes.

    A página 108 dá uma razão específica para a ``MultiHeadAttention`` ser mais
    eficiente: as projeções Q, K e V exigem **uma** multiplicação de matriz cada,
    em vez de uma por cabeça, e esse é o passo mais caro do mecanismo. Este bloco
    mede duas coisas diferentes, porque elas não respondem da mesma forma:

    - **a projeção isolada** — ``num_heads`` camadas lineares ``d × head_dim``
      contra uma única camada ``d × d``. É exatamente a afirmação do livro, medida
      sem interferência;
    - **a passagem direta completa** de cada módulo. Aqui a comparação não é
      simétrica: a ``MultiHeadAttention`` executa ainda a projeção de saída
      ``out_proj`` (uma multiplicação ``d × d`` que o wrapper não tem) e
      reorganiza os tensores com ``view``/``transpose``/``contiguous``. Esses
      custos extras podem anular — e, em CPU e dimensões grandes, superar — a
      economia obtida nas projeções.

    Separar as duas medidas evita atribuir à divisão de pesos um efeito que, no
    tempo total, vem de outra parte do módulo.
    """
    print("\n  Empilhamento x divisão de pesos, em escalas crescentes:")
    print(f"  (torch usando {torch.get_num_threads()} threads de CPU)")

    linhas: list[dict] = []
    for d, h, n in ESCALAS:
        dados = lote_de_embeddings(d_in=d, max_length=n, batch_size=4)
        x = dados["embeddings"]

        semear()
        wrapper = MultiHeadAttentionWrapper(d, d // h, n, 0.0, h).eval()
        semear()
        eficiente = MultiHeadAttention(d, d, n, 0.0, h).eval()

        # Passo isolado: calcular as keys de todas as cabeças.
        projecoes_separadas = [cabeca.W_key for cabeca in wrapper.heads]

        def projetar_separado() -> None:
            for camada in projecoes_separadas:
                camada(x)

        t_proj_wrapper = cronometrar(projetar_separado, repeticoes=60)
        t_proj_unica = cronometrar(lambda: eficiente.W_key(x), repeticoes=60)

        t_wrapper = cronometrar(lambda: wrapper(x), repeticoes=40)
        t_eficiente = cronometrar(lambda: eficiente(x), repeticoes=40)

        linhas.append(
            {
                "d_modelo": d,
                "num_heads": h,
                "num_tokens": n,
                "projecao_separada_ms": t_proj_wrapper,
                "projecao_unica_ms": t_proj_unica,
                "ganho_na_projecao": t_proj_wrapper / t_proj_unica,
                "forward_wrapper_ms": t_wrapper,
                "forward_multihead_ms": t_eficiente,
                "ganho_no_forward": t_wrapper / t_eficiente,
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp04-empilhamento-vs-divisao")
    print(
        "  ganho > 1 significa que a divisão de pesos é mais rápida; "
        "< 1, que o empilhamento é."
    )

    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 3.9))
    rotulos = [
        f"d={l['d_modelo']}\nh={l['num_heads']}, n={l['num_tokens']}" for l in linhas
    ]
    posicoes = list(range(len(linhas)))
    largura = 0.38

    paineis = [
        (
            eixos[0],
            "projecao_separada_ms",
            "projecao_unica_ms",
            f"Só as projeções de keys",
            "uma camada por cabeça",
            "uma única camada d × d",
            "ganho_na_projecao",
        ),
        (
            eixos[1],
            "forward_wrapper_ms",
            "forward_multihead_ms",
            "Passagem direta completa",
            "MultiHeadAttentionWrapper",
            "MultiHeadAttention",
            "ganho_no_forward",
        ),
    ]

    for ax, chave_a, chave_b, titulo, rotulo_a, rotulo_b, chave_ganho in paineis:
        ax.bar(
            [p - largura / 2 - 0.012 for p in posicoes],
            [l[chave_a] for l in linhas],
            width=largura,
            color=CATEGORICA[0],
            label=rotulo_a,
        )
        ax.bar(
            [p + largura / 2 + 0.012 for p in posicoes],
            [l[chave_b] for l in linhas],
            width=largura,
            color=CATEGORICA[1],
            label=rotulo_b,
        )
        for p, l in zip(posicoes, linhas):
            ax.text(
                p,
                max(l[chave_a], l[chave_b]) * 1.05,
                f"{l[chave_ganho]:.2f}x",
                ha="center",
                fontsize=8,
                color=TINTA_SECUNDARIA,
            )
        ax.set_xticks(posicoes, rotulos, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("milissegundos")
        ax.set_title(titulo, pad=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, max(max(l[chave_a], l[chave_b]) for l in linhas) * 1.28)

    fig.suptitle(
        "Experimento 04 — a economia da divisão de pesos está nas projeções; "
        "no tempo total ela disputa com a out_proj e as transposições",
        y=1.05,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp04-empilhamento-vs-divisao")
    return linhas


def _graficos(linhas: list[dict], pesos_mha) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 3.9))

    nomes = [l["mecanismo"] for l in linhas]
    posicoes = range(len(nomes))

    # Barras horizontais: rótulos longos ficam legíveis sem rotação.
    barras = eixos[0].barh(
        list(posicoes),
        [l["tempo_ms"] for l in linhas],
        color=CATEGORICA[0],
        height=0.62,
    )
    eixos[0].set_yticks(list(posicoes), nomes, fontsize=8)
    eixos[0].invert_yaxis()
    eixos[0].grid(axis="y", visible=False)
    eixos[0].set_xlabel("milissegundos")
    eixos[0].set_title("Tempo da passagem direta", pad=8)
    for barra, linha in zip(barras, linhas):
        eixos[0].text(
            barra.get_width() * 1.02,
            barra.get_y() + barra.get_height() / 2,
            f"{linha['tempo_ms']:.2f} ms · {linha['parametros']:,} par.",
            va="center",
            fontsize=7.5,
            color=TINTA_SECUNDARIA,
        )
    eixos[0].set_xlim(0, max(l["tempo_ms"] for l in linhas) * 1.45)

    # Uma cabeça por painel embutido: mostra que as matrizes diferem entre si.
    media = pesos_mha.detach().mean(dim=0)
    n = min(32, media.shape[-1])
    eixos[1].axis("off")
    eixos[1].set_title(
        f"As {media.shape[0]} cabeças da MultiHeadAttention\n"
        f"(primeiros {n} tokens da mesma sequência)",
        pad=8,
    )
    # A linha 0 vale sempre 1 (único token visível) e achataria a escala de cor
    # de todas as demais; a escala é fixada pelo resto da matriz.
    vmax = float(media[:, 2:n, :n].max())
    for k in range(media.shape[0]):
        sub = eixos[1].inset_axes([0.5 * (k % 2), 0.5 * (1 - k // 2), 0.44, 0.44])
        sub.imshow(media[k, :n, :n].numpy(), cmap=SEQUENCIAL, vmin=0, vmax=vmax)
        sub.set_title(f"cabeça {k}", fontsize=7.5, color=TINTA_SECUNDARIA, pad=3)
        sub.set_xticks([])
        sub.set_yticks([])
        sub.grid(False)
        for lado in sub.spines.values():
            lado.set_color(GRADE)

    fig.suptitle(
        "Experimento 04 — custo dos mecanismos e especialização das cabeças", y=1.03
    )
    fig.tight_layout()
    salvar_figura(fig, "exp04-self-vs-multihead")


if __name__ == "__main__":
    executar()

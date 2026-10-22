"""Experimento 08 — por que a posição precisa ser representada.

**O que varia:** a presença do embedding posicional (``usar_posicional``).
**O que fica fixo:** ``d_emb = 64``, ``context_length = 32``, semente 123,
tokenizador BPE.

**Pergunta:** o embedding de token, sozinho, é capaz de distinguir posições e
ordens diferentes?

Três demonstrações:

(a) **Mesmo token, posições diferentes.** Na frase *"Then the dog chased the
    cat, and the cat chased the dog."* o token ``" the"`` aparece quatro vezes.
    Sem embedding posicional as quatro ocorrências recebem vetores idênticos
    (similaridade de cosseno exatamente 1); com ele, passam a ser distintas.

(b) **Mesmas palavras, ordem diferente.** *"the dog bit the man"* e *"the man
    bit the dog"* têm os mesmos tokens. Sem posição, o *conjunto* de vetores das
    duas frases é idêntico — qualquer operação que não dependa da ordem das
    linhas (como, a Sprint 3 mostrou, a própria atenção) não as distingue. Com
    posição, ``" dog"`` na posição 1 e na posição 4 vira dois vetores diferentes.
    Uma ressalva medida aqui: a *média* dos vetores continua igual mesmo com
    posição, porque a soma é linear. A ordem só é aproveitada por operações que
    combinam token e posição de forma não linear — o softmax da atenção.

(c) **O que a posição significa antes do treinamento.** Os vetores posicionais
    começam aleatórios: a posição 3 é tão parecida com a 4 quanto com a 30. A
    noção de *proximidade* entre posições não está na inicialização — terá de
    ser aprendida no treinamento da Sprint 5.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from comum import (
    CATEGORICA,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
    cabecalho,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
    similaridade_cosseno,
)

from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

D_EMB = 64
CONTEXT_LENGTH = 32
FRASE = "Then the dog chased the cat, and the cat chased the dog."
ORDEM_A = "the dog bit the man"
ORDEM_B = "the man bit the dog"

# Divergente: laranja (negativo) -> superfície (zero) -> azul (positivo).
DIVERGENTE = LinearSegmentedColormap.from_list(
    "divergente", [CATEGORICA[1], SUPERFICIE, CATEGORICA[0]]
)


def _camadas(vocab_size: int) -> tuple[EmbeddingDeEntrada, EmbeddingDeEntrada]:
    """Mesmos pesos, com e sem a soma posicional."""
    semear()
    com = EmbeddingDeEntrada(vocab_size, D_EMB, CONTEXT_LENGTH)
    sem = EmbeddingDeEntrada(vocab_size, D_EMB, CONTEXT_LENGTH, usar_posicional=False)
    sem.load_state_dict(com.state_dict())
    return com.eval(), sem.eval()


def _pares_do_mesmo_token(ids: torch.Tensor) -> torch.Tensor:
    iguais = ids[:, None] == ids[None, :]
    return iguais & ~torch.eye(len(ids), dtype=torch.bool)


def executar() -> list[dict]:
    cabecalho("Experimento 08 — por que a posição precisa ser representada")
    bpe = TokenizadorBPE()
    com, sem = _camadas(bpe.tamanho_vocabulario)

    # (a) mesmo token em posições diferentes
    ids = torch.tensor(bpe.encode(FRASE))
    rotulos = [t.strip() or "·" for t in bpe.ids_para_tokens(ids)]
    with torch.no_grad():
        sim_sem = similaridade_cosseno(sem(ids))
        sim_com = similaridade_cosseno(com(ids))
        sim_pos = similaridade_cosseno(com.pos_emb.weight)
    mesmo_token = _pares_do_mesmo_token(ids)
    print(f"  frase: {FRASE!r}  ({len(ids)} tokens)")
    print(f"  pares de posições com o mesmo token: {int(mesmo_token.sum()) // 2}")

    # (b) mesmas palavras, ordem trocada: " dog" está na posição 1 de A e 4 de B
    ids_a = torch.tensor(bpe.encode(ORDEM_A))
    ids_b = torch.tensor(bpe.encode(ORDEM_B))
    dog = bpe.encode(" dog")[0]
    pos_a = int((ids_a == dog).nonzero()[0])
    pos_b = int((ids_b == dog).nonzero()[0])

    def mesmo_conjunto(camada: EmbeddingDeEntrada) -> bool:
        """As duas frases produzem o mesmo conjunto de vetores, em qualquer ordem?"""
        a = sorted(map(tuple, camada(ids_a).tolist()))
        b = sorted(map(tuple, camada(ids_b).tolist()))
        return a == b

    with torch.no_grad():
        dog_sem = float((sem(ids_a)[pos_a] - sem(ids_b)[pos_b]).norm())
        dog_com = float((com(ids_a)[pos_a] - com(ids_b)[pos_b]).norm())
        conjunto_sem = mesmo_conjunto(sem)
        conjunto_com = mesmo_conjunto(com)
        # A média é linear: média(token + posição) = média(token) + média(posição).
        # Com os mesmos tokens e as mesmas posições, ela não muda com a ordem.
        media_com = float((com(ids_a).mean(0) - com(ids_b).mean(0)).norm())

    # (c) proximidade entre posições na inicialização
    distancias = torch.arange(CONTEXT_LENGTH)
    distancia = (distancias[:, None] - distancias[None, :]).abs()
    vizinhas = float(sim_pos[distancia == 1].mean())
    distantes = float(sim_pos[distancia >= 16].mean())

    linhas = [
        {"medida": "cosseno entre ocorrências do mesmo token — sem posição",
         "valor": float(sim_sem[mesmo_token].mean())},
        {"medida": "cosseno entre ocorrências do mesmo token — com posição",
         "valor": float(sim_com[mesmo_token].mean())},
        {"medida": "distância entre o vetor de ' dog' em A e em B — sem posição",
         "valor": dog_sem},
        {"medida": "distância entre o vetor de ' dog' em A e em B — com posição",
         "valor": dog_com},
        {"medida": "A e B formam o mesmo conjunto de vetores — sem posição (1 = sim)",
         "valor": float(conjunto_sem)},
        {"medida": "A e B formam o mesmo conjunto de vetores — com posição (1 = sim)",
         "valor": float(conjunto_com)},
        {"medida": "‖média(A) − média(B)‖ — com posição (a média descarta a ordem)",
         "valor": media_com},
        {"medida": "cosseno médio entre posições vizinhas (|i−j| = 1)",
         "valor": vizinhas},
        {"medida": "cosseno médio entre posições distantes (|i−j| ≥ 16)",
         "valor": distantes},
        {"medida": "norma média do embedding de token",
         "valor": float(com.token_emb.weight.detach()[ids].norm(dim=-1).mean())},
        {"medida": "norma média do embedding posicional",
         "valor": float(com.pos_emb.weight.detach().norm(dim=-1).mean())},
    ]
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp08-embeddings-posicionais")
    _grafico(sim_sem, sim_com, sim_pos, rotulos)
    return linhas


def _mapa(ax: plt.Axes, matriz: torch.Tensor, titulo: str, rotulos: list[str] | None) -> None:
    im = ax.imshow(matriz.numpy(), cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_title(titulo, color=TINTA, pad=8)
    ax.grid(False)
    ax.tick_params(length=0)
    n = matriz.shape[0]
    if rotulos:
        ax.set_xticks(range(n), rotulos, rotation=90, fontsize=7)
        ax.set_yticks(range(n), rotulos, fontsize=7)
    else:
        ax.set_xlabel("posição")
        ax.set_ylabel("posição")
    barra = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    barra.set_label("similaridade de cosseno", color=TINTA_SECUNDARIA, fontsize=8)
    barra.ax.tick_params(labelsize=7)


def _grafico(sim_sem, sim_com, sim_pos, rotulos) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6))
    _mapa(eixos[0], sim_sem, "(a) só embedding de token", rotulos)
    _mapa(eixos[1], sim_com, "(a) token + posição", rotulos)
    _mapa(eixos[2], sim_pos, "(c) entre vetores posicionais\n(inicialização)", None)
    fig.suptitle(
        f"Experimento 08 — similaridade entre representações (d_emb = {D_EMB})", y=1.02
    )
    fig.tight_layout()
    salvar_figura(fig, "exp08-embeddings-posicionais")


if __name__ == "__main__":
    executar()

"""Utilidades compartilhadas pelos experimentos da Sprint 6.

Centraliza o acesso aos módulos de ``src/``, a semente, o dispositivo, o
treino parametrizável do classificador, as métricas sobre o texto gerado e a
gravação de tabelas e figuras.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint06/``.
Os pesos pré-treinados e os checkpoints do fine-tuning ficam em ``modelos/``,
que não é versionada.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

# --- caminhos --------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"
RESULTADOS = RAIZ / "results-by-sprints" / "sprint06"
RESULTADOS.mkdir(parents=True, exist_ok=True)
CHECKPOINT_SPRINT5 = RAIZ / "results-by-sprints" / "sprint05" / "base-modelo.pt"
CHECKPOINT_CLASSIFICADOR = RAIZ / "modelos" / "classificador-spam.pt"
CHECKPOINT_INSTRUCOES = RAIZ / "modelos" / "instrucoes-gpt2.pt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classificacao import (  # noqa: E402
    acuracia_do_loader,
    adaptar_para_classificacao,
    contar_treinaveis,
    criar_loaders_classificacao,
    preparar_base_spam,
    treinar_classificador,
)
from pesos_pretreinados import carregar_gpt2_pretreinado  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402

# --- reprodutibilidade e dispositivo ----------------------------------------
SEMENTE = 123
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXTO = "Every effort moves you"
TOKENIZADOR = TokenizadorBPE()


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


def cabecalho(titulo: str) -> None:
    print("\n" + "=" * 74)
    print(titulo)
    print("=" * 74)


# --- fine-tuning de classificação --------------------------------------------
# Configuração base do treino do classificador (seção 6.7 do livro).
TREINO_CLASSIFICADOR = {
    "modelo": "GPT-2 124M pré-treinado",
    "num_epocas": 5,
    "batch_size": 8,
    "otimizador": "AdamW",
    "taxa_aprendizado": 5e-5,
    "weight_decay": 0.1,
    "blocos_treinaveis": 1,        # None = modelo inteiro; 0 = só a cabeça
    "treinar_norm_final": True,
    "posicao": -1,                 # token cuja saída é usada na classificação
    "max_length": None,            # None = comprimento da maior mensagem do treino
    "frequencia_avaliacao": 50,
    "lotes_avaliacao": 5,
    "semente": SEMENTE,
}


def novo_classificador(blocos_treinaveis: int | None = 1, treinar_norm_final: bool = True,
                       semente: int = SEMENTE):
    """GPT-2 124M pré-treinado com a cabeça de classificação recém-criada.

    A semente é fixada antes de criar a cabeça, a única parte com pesos
    aleatórios: duas chamadas produzem o mesmo ponto de partida.
    """
    modelo = carregar_gpt2_pretreinado("small", dispositivo=DISPOSITIVO)
    semear(semente)
    return adaptar_para_classificacao(modelo, 2, blocos_treinaveis, treinar_norm_final)


def treinar_configuracao(verboso: bool = False, **alteracoes) -> dict:
    """Ajusta o classificador com ``TREINO_CLASSIFICADOR`` + ``alteracoes``.

    Devolve o modelo, o histórico, a configuração efetiva, o tempo de treino e
    as acurácias finais nos conjuntos **inteiros** de validação e de teste.
    """
    cfg = dict(TREINO_CLASSIFICADOR, **alteracoes)
    preparar_base_spam(semente=SEMENTE)
    lt, lv, lte = criar_loaders_classificacao(
        tokenizador=TOKENIZADOR, batch_size=cfg["batch_size"], max_length=cfg["max_length"],
        semente=cfg["semente"])
    cfg["max_length"] = lt.dataset.max_length
    modelo = novo_classificador(cfg["blocos_treinaveis"], cfg["treinar_norm_final"], cfg["semente"])
    treinaveis, total = contar_treinaveis(modelo)
    otimizador = getattr(torch.optim, cfg["otimizador"])(
        [p for p in modelo.parameters() if p.requires_grad],
        lr=cfg["taxa_aprendizado"], weight_decay=cfg["weight_decay"])

    inicio = time.perf_counter()
    historico = treinar_classificador(
        modelo, lt, lv, otimizador, DISPOSITIVO, cfg["num_epocas"],
        cfg["frequencia_avaliacao"], cfg["lotes_avaliacao"], cfg["posicao"],
        verboso=verboso, lotes_acuracia=len(lv))
    tempo = time.perf_counter() - inicio

    return {
        "modelo": modelo,
        "otimizador": otimizador,
        "historico": historico,
        "cfg": cfg,
        "loaders": (lt, lv, lte),
        "tempo_s": tempo,
        "treinaveis": treinaveis,
        "parametros": total,
        "acuracia_validacao": acuracia_do_loader(lv, modelo, DISPOSITIVO, posicao=cfg["posicao"]),
        "acuracia_teste": acuracia_do_loader(lte, modelo, DISPOSITIVO, posicao=cfg["posicao"]),
    }


# --- fine-tuning de instruções -----------------------------------------------
# Configuração do treino de instruções (seção 7.6 do livro), com o modelo e o
# número de blocos treináveis reduzidos para caber em CPU com 7,5 GB de RAM.
TREINO_INSTRUCOES = {
    "tamanho": "small",            # GPT-2 124M; o capítulo usa o "medium" (355M)
    "num_epocas": 2,
    "batch_size": 8,
    "otimizador": "AdamW",
    "taxa_aprendizado": 5e-5,
    "weight_decay": 0.1,
    "blocos_treinaveis": 6,        # None = modelo inteiro, como no capítulo
    "frequencia_avaliacao": 10,
    "lotes_avaliacao": 5,
    "semente": SEMENTE,
}


# --- métricas sobre o texto gerado ------------------------------------------
def distinct_n(ids: list[int], n: int) -> float:
    """Fração de n-gramas distintos entre os gerados: 1 = nenhuma repetição."""
    ngramas = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def repeticao_n(ids: list[int], n: int = 3) -> float:
    """Fração dos n-gramas que já tinham aparecido antes no mesmo texto."""
    vistos, repetidos, total = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        repetidos += g in vistos
        vistos.add(g)
        total += 1
    return repetidos / total if total else 0.0


# --- gravação ---------------------------------------------------------------
def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        for linha in linhas:
            w.writerow({k: _fmt(v) for k, v in linha.items()})
    print(f"  tabela -> results-by-sprints/sprint06/{destino.name}")
    return destino


def imprimir_tabela(linhas: list[dict]) -> None:
    colunas = list(linhas[0])
    larguras = [max(len(c), max(len(_fmt(l[c])) for l in linhas)) for c in colunas]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  " + "  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  " + "  ".join(_fmt(l[c]).ljust(w) for c, w in zip(colunas, larguras)))


def salvar_figura(fig, nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"  figura -> results-by-sprints/sprint06/{destino.name}")
    return destino


__all__ = [
    "RAIZ", "RESULTADOS", "CHECKPOINT_SPRINT5", "CHECKPOINT_CLASSIFICADOR", "SEMENTE",
    "CHECKPOINT_INSTRUCOES", "TREINO_INSTRUCOES",
    "TREINO_CLASSIFICADOR", "novo_classificador", "treinar_configuracao", "DISPOSITIVO", "CONTEXTO", "TOKENIZADOR",
    "semear", "cabecalho", "distinct_n", "repeticao_n",
    "salvar_tabela", "imprimir_tabela", "salvar_figura", "plt",
]

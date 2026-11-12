"""Utilidades compartilhadas pelos experimentos da Sprint 5.

Centraliza o que todos os experimentos de treinamento precisam: acesso aos
módulos de ``src/``, semente, dispositivo, os loaders de treino e validação, um
treinamento "rápido" parametrizável, métricas sobre o texto gerado e a
gravação de tabelas e figuras.

Os experimentos de varredura (E1–E4) usam um **GPT reduzido** — mesmo
vocabulário e contexto do treino base, mas ``emb_dim = 256``, 4 cabeças e 4
blocos —, para que dezenas de treinamentos caibam em poucos minutos. Os
experimentos de geração (E5–E6) usam o checkpoint do treino base
(``treinar_base.py``), o GPT-2 124M.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint05/``.
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
RESULTADOS = RAIZ / "results-by-sprints" / "sprint05"
RESULTADOS.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE = RESULTADOS / "base-modelo.pt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dados import carregar_texto, criar_loaders_treino_validacao, dividir_treino_validacao  # noqa: E402
from gpt import GPTModel, config_gpt2, contar_parametros  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402
from treinamento import ConfigTreinamento, avaliar_modelo, treinar_modelo  # noqa: E402

# --- reprodutibilidade e dispositivo ----------------------------------------
SEMENTE = 123
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXTO = "Every effort moves you"
TOKENIZADOR = TokenizadorBPE()


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


# --- configuração-base das varreduras ---------------------------------------
CONFIG_REDUZIDO = config_gpt2("small", context_length=256, emb_dim=256, n_heads=4, n_layers=4)
TREINO_BASE = ConfigTreinamento(dispositivo=DISPOSITIVO)


def cabecalho(titulo: str) -> None:
    print("\n" + "=" * 74)
    print(titulo)
    print("=" * 74)


# --- treinamento parametrizável ---------------------------------------------
def treinar_rapido(
    cfg_modelo: dict | None = None,
    cfg_treino: ConfigTreinamento | None = None,
    contexto_inicial: str | None = None,
    verboso: bool = False,
) -> dict:
    """Treina um modelo do zero e devolve histórico, perdas finais e custo.

    Usa os mesmos dados e a mesma divisão 90/10 do treino base. A semente é
    fixada antes dos loaders e antes do modelo, de modo que duas chamadas com
    a mesma configuração produzem o mesmo resultado.
    """
    cfg_modelo = cfg_modelo or CONFIG_REDUZIDO
    cfg_treino = cfg_treino or TREINO_BASE
    dispositivo = torch.device(cfg_treino.dispositivo)

    semear(cfg_treino.semente)
    texto = carregar_texto(RAIZ / cfg_treino.corpus)
    lt, lv = criar_loaders_treino_validacao(
        texto, cfg_treino.proporcao_treino, cfg_treino.batch_size,
        cfg_treino.max_length, cfg_treino.stride, TOKENIZADOR,
    )
    semear(cfg_treino.semente)
    modelo = GPTModel(cfg_modelo).to(dispositivo)
    otimizador = cfg_treino.criar_otimizador(modelo)

    saida = sys.stdout
    if not verboso:
        sys.stdout = _Silencio()
    try:
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        inicio = time.perf_counter()
        historico = treinar_modelo(
            modelo, lt, lv, otimizador, dispositivo, cfg_treino.num_epocas,
            cfg_treino.frequencia_avaliacao, cfg_treino.lotes_avaliacao,
            contexto_inicial=contexto_inicial, tokenizador=TOKENIZADOR,
        )
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        tempo = time.perf_counter() - inicio
    finally:
        sys.stdout = saida

    perda_treino, perda_validacao = avaliar_modelo(modelo, lt, lv, dispositivo)
    return {
        "modelo": modelo,
        "historico": historico,
        "perda_treino": perda_treino,
        "perda_validacao": perda_validacao,
        "melhor_validacao": min(historico["perda_validacao"]),
        "tempo_s": tempo,
        "parametros": contar_parametros(modelo),
        "tokens_vistos": historico["tokens_vistos"][-1] if historico["tokens_vistos"] else 0,
    }


class _Silencio:
    def write(self, _texto):
        pass

    def flush(self):
        pass


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


_NGRAMAS_TREINO: dict[int, set] = {}


def copia_do_treino(ids: list[int], n: int = 8) -> float:
    """Fração dos n-gramas gerados que aparecem literalmente no texto de treinamento.

    Mede memorização: perto de 1, o modelo está reproduzindo o corpus.
    """
    if n not in _NGRAMAS_TREINO:
        treino, _ = dividir_treino_validacao(carregar_texto(), TREINO_BASE.proporcao_treino)
        t = TOKENIZADOR.encode(treino)
        _NGRAMAS_TREINO[n] = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    ngramas = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return sum(g in _NGRAMAS_TREINO[n] for g in ngramas) / len(ngramas) if ngramas else 0.0


# --- gravação ---------------------------------------------------------------
def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        for linha in linhas:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in linha.items()})
    print(f"  tabela -> results-by-sprints/sprint05/{destino.name}")
    return destino


def imprimir_tabela(linhas: list[dict]) -> None:
    colunas = list(linhas[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    larguras = [max(len(c), max(len(fmt(l[c])) for l in linhas)) for c in colunas]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  " + "  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  " + "  ".join(fmt(l[c]).ljust(w) for c, w in zip(colunas, larguras)))


def salvar_figura(fig, nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"  figura -> results-by-sprints/sprint05/{destino.name}")
    return destino


__all__ = [
    "RAIZ", "RESULTADOS", "CHECKPOINT_BASE", "SEMENTE", "DISPOSITIVO", "CONTEXTO", "TOKENIZADOR",
    "CONFIG_REDUZIDO", "TREINO_BASE", "semear", "cabecalho", "treinar_rapido",
    "distinct_n", "repeticao_n", "copia_do_treino",
    "salvar_tabela", "imprimir_tabela", "salvar_figura", "plt",
]

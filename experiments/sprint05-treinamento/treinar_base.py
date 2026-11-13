"""Treinamento base do GPT — Sprint 5 (seção 5.2).

Reproduz o treinamento do capítulo: GPT-2 124M com contexto de 256 tokens,
treinado por 10 épocas sobre ``the-verdict.txt`` com AdamW. Mede as perdas de
treinamento e validação antes, durante e depois do treino, gera uma amostra de
texto por época e grava o checkpoint final.

Artefatos em ``results-by-sprints/sprint05/``:

- ``base-config.json``       configuração do modelo e do treinamento;
- ``base-historico.csv``     perdas e perplexidades a cada avaliação;
- ``base-amostras.txt``      amostra greedy ao fim de cada época;
- ``base-perdas.png``        curvas de perda (treino × validação) e perplexidade;
- ``base-execucao.log``      registro do console, com o ambiente;
- ``base-modelo.pt``         checkpoint (não versionado, ver ``.gitignore``).

Uso, a partir da raiz do repositório::

    uv run python experiments/sprint05-treinamento/treinar_base.py
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from checkpoint import salvar_checkpoint  # noqa: E402
from dados import carregar_texto, criar_loaders_treino_validacao  # noqa: E402
from gpt import GPTModel, config_gpt2, contar_parametros  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402
from treinamento import (  # noqa: E402
    ConfigTreinamento,
    avaliar_modelo,
    perplexidade,
    treinar_modelo,
)

RESULTADOS = RAIZ / "results-by-sprints" / "sprint05"
CONTEXTO = "Every effort moves you"


class _Espelho:
    """Escreve simultaneamente no console e no arquivo de log."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for d in self.destinos:
            d.write(texto)

    def flush(self):
        for d in self.destinos:
            d.flush()


def treinar(cfg_treino: ConfigTreinamento, cfg_modelo: dict, prefixo: str = "base") -> dict:
    """Treina, avalia e grava todos os artefatos com o ``prefixo`` dado."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device(cfg_treino.dispositivo)
    tokenizador = TokenizadorBPE()

    torch.manual_seed(cfg_treino.semente)
    texto = carregar_texto(RAIZ / cfg_treino.corpus)
    loader_treino, loader_validacao = criar_loaders_treino_validacao(
        texto,
        proporcao_treino=cfg_treino.proporcao_treino,
        batch_size=cfg_treino.batch_size,
        max_length=cfg_treino.max_length,
        stride=cfg_treino.stride,
        tokenizador=tokenizador,
    )

    torch.manual_seed(cfg_treino.semente)
    modelo = GPTModel(cfg_modelo).to(dispositivo)
    otimizador = cfg_treino.criar_otimizador(modelo)

    print(f"  parâmetros     : {contar_parametros(modelo):,}")
    print(f"  lotes treino   : {len(loader_treino)} | lotes validação: {len(loader_validacao)}")
    print(f"  configuração   : {json.dumps(cfg_treino.para_dict(), ensure_ascii=False)}")

    inicial_treino, inicial_validacao = avaliar_modelo(
        modelo, loader_treino, loader_validacao, dispositivo
    )
    print(
        f"\n  antes do treino: perda treino {inicial_treino:.3f} "
        f"(perplexidade {perplexidade(inicial_treino):,.0f}) | "
        f"validação {inicial_validacao:.3f} (perplexidade {perplexidade(inicial_validacao):,.0f})\n"
    )

    inicio = time.perf_counter()
    historico = treinar_modelo(
        modelo, loader_treino, loader_validacao, otimizador, dispositivo,
        num_epocas=cfg_treino.num_epocas,
        frequencia_avaliacao=cfg_treino.frequencia_avaliacao,
        lotes_avaliacao=cfg_treino.lotes_avaliacao,
        contexto_inicial=CONTEXTO,
        tokenizador=tokenizador,
    )
    duracao = time.perf_counter() - inicio

    final_treino, final_validacao = avaliar_modelo(
        modelo, loader_treino, loader_validacao, dispositivo
    )
    resumo = {
        "perda_treino_inicial": inicial_treino,
        "perda_validacao_inicial": inicial_validacao,
        "perda_treino_final": final_treino,
        "perda_validacao_final": final_validacao,
        "perplexidade_treino_final": perplexidade(final_treino),
        "perplexidade_validacao_final": perplexidade(final_validacao),
        "tempo_treino_s": duracao,
        "tokens_vistos": historico["tokens_vistos"][-1],
    }
    print(
        f"\n  depois do treino: perda treino {final_treino:.3f} "
        f"(perplexidade {perplexidade(final_treino):,.1f}) | "
        f"validação {final_validacao:.3f} (perplexidade {perplexidade(final_validacao):,.1f})"
    )
    print(f"  tempo de treino: {duracao:.1f} s")

    _gravar(prefixo, cfg_treino, cfg_modelo, historico, resumo)
    salvar_checkpoint(
        RESULTADOS / f"{prefixo}-modelo.pt", modelo, otimizador,
        cfg_treino.para_dict(), historico,
    )
    print(f"  checkpoint -> results-by-sprints/sprint05/{prefixo}-modelo.pt")
    return {"historico": historico, "resumo": resumo}


def _gravar(prefixo, cfg_treino, cfg_modelo, historico, resumo) -> None:
    (RESULTADOS / f"{prefixo}-config.json").write_text(
        json.dumps({"modelo": cfg_modelo, "treino": cfg_treino.para_dict(), "resumo": resumo},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(RESULTADOS / f"{prefixo}-historico.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passo", "epoca", "tokens_vistos", "perda_treino", "perda_validacao",
                    "perplexidade_treino", "perplexidade_validacao"])
        for i in range(len(historico["passo"])):
            pt, pv = historico["perda_treino"][i], historico["perda_validacao"][i]
            w.writerow([historico["passo"][i], historico["epoca"][i], historico["tokens_vistos"][i],
                        f"{pt:.4f}", f"{pv:.4f}", f"{perplexidade(pt):.2f}", f"{perplexidade(pv):.2f}"])

    with open(RESULTADOS / f"{prefixo}-amostras.txt", "w", encoding="utf-8") as f:
        f.write(f"Contexto inicial: {CONTEXTO!r} (greedy, 50 tokens)\n\n")
        for epoca, amostra in enumerate(historico["amostras"], start=1):
            f.write(f"Época {epoca:>2}: {amostra!r}\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tokens = historico["tokens_vistos"]
    a1.plot(tokens, historico["perda_treino"], label="treino")
    a1.plot(tokens, historico["perda_validacao"], "--", label="validação")
    a1.set_xlabel("tokens vistos")
    a1.set_ylabel("perda (entropia cruzada)")
    a1.set_title("Perda durante o treinamento")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(tokens, [perplexidade(p) for p in historico["perda_treino"]], label="treino")
    a2.plot(tokens, [perplexidade(p) for p in historico["perda_validacao"]], "--", label="validação")
    a2.set_yscale("log")
    a2.set_xlabel("tokens vistos")
    a2.set_ylabel("perplexidade (escala log)")
    a2.set_title("Perplexidade durante o treinamento")
    a2.legend()
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"{prefixo}-perdas.png", dpi=150)
    plt.close(fig)
    print(f"  artefatos -> results-by-sprints/sprint05/{prefixo}-*")


def main() -> None:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_treino = ConfigTreinamento(dispositivo=dispositivo)
    cfg_modelo = config_gpt2("small", context_length=256)

    with open(RESULTADOS / "base-execucao.log", "w", encoding="utf-8") as log:
        antigo = sys.stdout
        sys.stdout = _Espelho(antigo, log)
        try:
            print("=" * 72)
            print("Sprint 5 — treinamento base do GPT-2 124M (contexto 256)")
            print("=" * 72)
            print(f"  python         : {platform.python_version()}")
            print(f"  torch          : {torch.__version__}")
            print(f"  dispositivo    : {dispositivo}"
                  + (f" ({torch.cuda.get_device_name(0)})" if dispositivo == "cuda" else ""))
            print(f"  sistema        : {platform.platform()}")
            treinar(cfg_treino, cfg_modelo)
        finally:
            sys.stdout = antigo


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

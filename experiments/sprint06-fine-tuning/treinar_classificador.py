"""Treino base do classificador de spam — Sprint 6 (seção 6.7).

Parte do GPT-2 124M pré-treinado, troca a cabeça de saída por uma de duas
classes, deixa treináveis o último bloco Transformer, a norma final e a cabeça
nova, e ajusta por 5 épocas com AdamW (``lr = 5e-5``, ``weight_decay = 0.1``,
lotes de 8) sobre a base de spam balanceada.

Artefatos em ``results-by-sprints/sprint06/``:

- ``classificador-config.json``     configuração do modelo e do treino, tamanhos
  dos conjuntos, parâmetros treináveis, tempo e acurácias finais;
- ``classificador-historico.csv``   perdas a cada avaliação;
- ``classificador-acuracia.csv``    acurácias de treino e validação por época;
- ``classificador-curvas.png``      curvas de perda e de acurácia;
- ``classificador-execucao.log``    registro do console, com o ambiente.

O checkpoint vai para ``modelos/classificador-spam.pt`` (não versionado).

Uso, a partir da raiz do repositório::

    uv run python experiments/sprint06-fine-tuning/treinar_classificador.py
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime

import torch

from comum import (
    CHECKPOINT_CLASSIFICADOR,
    DISPOSITIVO,
    RESULTADOS,
    cabecalho,
    plt,
    salvar_figura,
    treinar_configuracao,
)
from checkpoint import salvar_checkpoint
from classificacao import acuracia_do_loader, perda_classificacao_do_loader


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


def treinar() -> dict:
    cabecalho("Sprint 6 — treino base do classificador de spam")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python / torch : {platform.python_version()} / {torch.__version__}")
    gpu = f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""
    print(f"  dispositivo    : {DISPOSITIVO}{gpu}\n")

    r = treinar_configuracao(verboso=True)
    modelo, cfg, historico = r["modelo"], r["cfg"], r["historico"]
    lt, lv, lte = r["loaders"]

    print("\n  avaliação final nos conjuntos inteiros...")
    acuracia_treino = acuracia_do_loader(lt, modelo, DISPOSITIVO)
    resumo = {
        "treino": cfg,
        "modelo": dict(modelo.cfg),
        "exemplos": {"treino": len(lt.dataset), "validacao": len(lv.dataset), "teste": len(lte.dataset)},
        "parametros_treinaveis": r["treinaveis"],
        "parametros_totais": r["parametros"],
        "dispositivo": DISPOSITIVO,
        "tempo_de_treino_s": round(r["tempo_s"], 1),
        "acuracia_treino": acuracia_treino,
        "acuracia_validacao": r["acuracia_validacao"],
        "acuracia_teste": r["acuracia_teste"],
        "perda_validacao": perda_classificacao_do_loader(lv, modelo, DISPOSITIVO),
        "perda_teste": perda_classificacao_do_loader(lte, modelo, DISPOSITIVO),
    }
    print(f"  parâmetros treináveis: {r['treinaveis']:,} de {r['parametros']:,}")
    print(f"  tempo de treino      : {r['tempo_s'] / 60:.1f} min")
    print(f"  acurácia             : treino {acuracia_treino:.2%} | "
          f"validação {r['acuracia_validacao']:.2%} | teste {r['acuracia_teste']:.2%}")

    (RESULTADOS / "classificador-config.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(RESULTADOS / "classificador-historico.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passo", "exemplos_vistos", "perda_treino", "perda_validacao"])
        for linha in zip(historico["passos"], historico["exemplos_vistos"],
                         historico["perda_treino"], historico["perda_validacao"]):
            w.writerow([linha[0], linha[1], f"{linha[2]:.4f}", f"{linha[3]:.4f}"])
    with open(RESULTADOS / "classificador-acuracia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "acuracia_treino", "acuracia_validacao"])
        for i, (a, b) in enumerate(zip(historico["acuracia_treino"], historico["acuracia_validacao"]), 1):
            w.writerow([i, f"{a:.4f}", f"{b:.4f}"])

    salvar_checkpoint(CHECKPOINT_CLASSIFICADOR, modelo, cfg_treino=cfg, historico=historico)
    print(f"  checkpoint -> modelos/{CHECKPOINT_CLASSIFICADOR.name}")
    _grafico(historico, resumo)
    return resumo


def _grafico(historico: dict, resumo: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(historico["exemplos_vistos"], historico["perda_treino"], label="treino")
    a1.plot(historico["exemplos_vistos"], historico["perda_validacao"], "--", label="validação")
    a1.set_xlabel("exemplos vistos")
    a1.set_ylabel("entropia cruzada")
    a1.set_title("Perda durante o fine-tuning")
    a1.grid(alpha=0.3)
    a1.legend()

    epocas = range(1, len(historico["acuracia_treino"]) + 1)
    a2.plot(epocas, historico["acuracia_treino"], "o-", label="treino (amostra)")
    a2.plot(epocas, historico["acuracia_validacao"], "s--", label="validação")
    a2.axhline(resumo["acuracia_teste"], color="gray", linestyle=":",
               label=f"teste, ao final ({resumo['acuracia_teste']:.1%})")
    a2.axhline(0.5, color="lightgray", linewidth=1)
    a2.set_xticks(list(epocas))
    a2.set_xlabel("época")
    a2.set_ylabel("acurácia")
    a2.set_ylim(0.45, 1.01)
    a2.set_title("Acurácia por época")
    a2.grid(alpha=0.3)
    a2.legend()
    salvar_figura(fig, "classificador-curvas")


if __name__ == "__main__":
    with open(RESULTADOS / "classificador-execucao.log", "w", encoding="utf-8") as log:
        saida = sys.stdout
        sys.stdout = _Espelho(saida, log)
        try:
            treinar()
        finally:
            sys.stdout = saida

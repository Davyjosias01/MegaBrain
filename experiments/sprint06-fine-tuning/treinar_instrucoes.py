"""Fine-tuning de instruções — Sprint 6 (seção 7.6).

Parte do GPT-2 124M pré-treinado e continua o treinamento, com o mesmo laço da
Sprint 5 (``treinar_modelo``), sobre os prompts de instrução formatados: 913
exemplos, 2 épocas, AdamW (``lr = 5e-5``, ``weight_decay = 0.1``), lotes de 8.

Diferenças em relação ao capítulo, ambas impostas pela máquina usada (CPU,
7,5 GB de RAM):

- o modelo é o GPT-2 **124M**, e não o 355M;
- só os ``BLOCOS_TREINAVEIS`` últimos blocos e a norma final são ajustados; o
  capítulo ajusta o modelo inteiro.

Artefatos em ``results-by-sprints/sprint06/``:

- ``instrucoes-config.json``     configuração, tamanhos dos conjuntos, perdas
  inicial e final e tempo;
- ``instrucoes-historico.csv``   perdas a cada avaliação;
- ``instrucoes-amostras.txt``    resposta gerada ao fim de cada época para a
  mesma instrução de validação;
- ``instrucoes-curvas.png``      curvas de perda;
- ``instrucoes-execucao.log``    registro do console, com o ambiente.

O checkpoint vai para ``modelos/instrucoes-gpt2.pt`` (não versionado).

Uso, a partir da raiz do repositório::

    uv run python experiments/sprint06-fine-tuning/treinar_instrucoes.py
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime

import torch

from comum import (
    CHECKPOINT_INSTRUCOES,
    DISPOSITIVO,
    RESULTADOS,
    SEMENTE,
    TOKENIZADOR,
    TREINO_INSTRUCOES,
    cabecalho,
    plt,
    salvar_figura,
    semear,
)
from checkpoint import salvar_checkpoint
from instrucoes import (
    criar_loaders_instrucoes,
    definir_treinaveis,
    montar_prompt,
    preparar_instrucoes,
)
from pesos_pretreinados import carregar_gpt2_pretreinado
from treinamento import perda_do_loader, perplexidade, treinar_modelo


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
    cfg = dict(TREINO_INSTRUCOES)
    cabecalho("Sprint 6 — fine-tuning de instruções")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python / torch : {platform.python_version()} / {torch.__version__}")
    gpu = f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""
    print(f"  dispositivo    : {DISPOSITIVO}{gpu}")

    treino, validacao, teste = preparar_instrucoes()
    modelo = carregar_gpt2_pretreinado(cfg["tamanho"], dispositivo=DISPOSITIVO)
    treinaveis, total = definir_treinaveis(modelo, cfg["blocos_treinaveis"])
    semear(cfg["semente"])
    lt, lv, _ = criar_loaders_instrucoes(
        treino, validacao, teste, TOKENIZADOR, cfg["batch_size"],
        max_length=modelo.cfg["context_length"], semente=cfg["semente"])
    otimizador = getattr(torch.optim, cfg["otimizador"])(
        [p for p in modelo.parameters() if p.requires_grad],
        lr=cfg["taxa_aprendizado"], weight_decay=cfg["weight_decay"])

    print(f"  exemplos       : treino {len(treino)} | validação {len(validacao)} | teste {len(teste)}")
    print(f"  parâmetros     : {treinaveis:,} treináveis de {total:,}")
    print(f"  configuração   : {json.dumps(cfg, ensure_ascii=False)}")

    inicial_treino = perda_do_loader(lt, modelo, DISPOSITIVO, cfg["lotes_avaliacao"])
    inicial_validacao = perda_do_loader(lv, modelo, DISPOSITIVO)
    print(f"\n  antes do ajuste: perda treino {inicial_treino:.3f} | "
          f"validação {inicial_validacao:.3f} (perplexidade {perplexidade(inicial_validacao):.1f})\n")

    contexto = montar_prompt(validacao[0])
    inicio = time.perf_counter()
    historico = treinar_modelo(
        modelo, lt, lv, otimizador, DISPOSITIVO, cfg["num_epocas"],
        cfg["frequencia_avaliacao"], cfg["lotes_avaliacao"],
        contexto_inicial=contexto, tokenizador=TOKENIZADOR, tokens_amostra=40)
    tempo = time.perf_counter() - inicio
    modelo.eval()

    final_treino = perda_do_loader(lt, modelo, DISPOSITIVO, cfg["lotes_avaliacao"])
    final_validacao = perda_do_loader(lv, modelo, DISPOSITIVO)
    print(f"\n  depois do ajuste: perda treino {final_treino:.3f} | "
          f"validação {final_validacao:.3f} (perplexidade {perplexidade(final_validacao):.1f})")
    print(f"  tempo de treino : {tempo / 60:.1f} min")

    resumo = {
        "treino": cfg,
        "modelo": dict(modelo.cfg),
        "exemplos": {"treino": len(treino), "validacao": len(validacao), "teste": len(teste)},
        "parametros_treinaveis": treinaveis,
        "parametros_totais": total,
        "dispositivo": DISPOSITIVO,
        "tempo_de_treino_s": round(tempo, 1),
        "perda_treino_inicial": inicial_treino,
        "perda_validacao_inicial": inicial_validacao,
        "perda_treino_final": final_treino,
        "perda_validacao_final": final_validacao,
    }
    (RESULTADOS / "instrucoes-config.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(RESULTADOS / "instrucoes-historico.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passo", "epoca", "tokens_vistos", "perda_treino", "perda_validacao"])
        for i in range(len(historico["passo"])):
            w.writerow([historico["passo"][i], historico["epoca"][i], historico["tokens_vistos"][i],
                        f"{historico['perda_treino'][i]:.4f}", f"{historico['perda_validacao'][i]:.4f}"])
    with open(RESULTADOS / "instrucoes-amostras.txt", "w", encoding="utf-8") as f:
        f.write("Resposta gerada ao fim de cada época (greedy, 40 tokens) para a instrução:\n\n")
        f.write(contexto + "\n\n")
        f.write(f"Resposta esperada: {validacao[0]['output']}\n\n")
        for epoca, amostra in enumerate(historico["amostras"], start=1):
            f.write(f"Época {epoca}: {amostra[len(contexto):]!r}\n")

    salvar_checkpoint(CHECKPOINT_INSTRUCOES, modelo, cfg_treino=cfg, historico=historico)
    print(f"  checkpoint -> modelos/{CHECKPOINT_INSTRUCOES.name}")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(historico["tokens_vistos"], historico["perda_treino"], label="treino")
    ax.plot(historico["tokens_vistos"], historico["perda_validacao"], "--", label="validação")
    ax.axhline(inicial_validacao, color="gray", linestyle=":",
               label=f"validação antes do ajuste ({inicial_validacao:.2f})")
    ax.set_xlabel("tokens vistos")
    ax.set_ylabel("entropia cruzada")
    ax.set_title("Perda durante o fine-tuning de instruções")
    ax.grid(alpha=0.3)
    ax.legend()
    salvar_figura(fig, "instrucoes-curvas")
    return resumo


if __name__ == "__main__":
    with open(RESULTADOS / "instrucoes-execucao.log", "w", encoding="utf-8") as log:
        saida = sys.stdout
        sys.stdout = _Espelho(saida, log)
        try:
            treinar()
        finally:
            sys.stdout = saida

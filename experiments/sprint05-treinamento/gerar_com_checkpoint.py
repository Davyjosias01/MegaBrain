"""Inferência a partir de um checkpoint salvo — Sprint 5 (seção 5.4).

Demonstra que o modelo carregado do disco gera texto usando apenas os
parâmetros salvos, sem treinar nada: a arquitetura vem da ``cfg_modelo``
gravada no checkpoint e os pesos do ``state_dict``.

Uso, a partir da raiz do repositório (depois de ``treinar_base.py``)::

    uv run python experiments/sprint05-treinamento/gerar_com_checkpoint.py
    uv run python experiments/sprint05-treinamento/gerar_com_checkpoint.py \\
        --checkpoint results-by-sprints/sprint05/base-modelo.pt \\
        --contexto "Every effort moves you" --tokens 50 --temperatura 1.4 --top-k 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from checkpoint import carregar_checkpoint  # noqa: E402
from geracao import gerar, ids_para_texto, texto_para_ids  # noqa: E402
from gpt import contar_parametros  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402

PADRAO = RAIZ / "results-by-sprints" / "sprint05" / "base-modelo.pt"


def gerar_do_checkpoint(
    caminho: Path | str,
    contexto: str,
    max_novos_tokens: int = 50,
    temperatura: float = 0.0,
    top_k: int | None = None,
    semente: int = 123,
    dispositivo: str | None = None,
) -> tuple[str, dict]:
    """Carrega o checkpoint e gera texto. Devolve ``(texto, dados_do_checkpoint)``."""
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo, dados = carregar_checkpoint(caminho, dispositivo)
    tokenizador = TokenizadorBPE()
    torch.manual_seed(semente)
    ids = texto_para_ids(contexto, tokenizador).to(dispositivo)
    with torch.no_grad():
        saida = gerar(
            modelo, ids, max_novos_tokens=max_novos_tokens,
            tamanho_contexto=modelo.cfg["context_length"],
            temperatura=temperatura, top_k=top_k,
        )
    dados["_parametros"] = contar_parametros(modelo)
    return ids_para_texto(saida, tokenizador), dados


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=str(PADRAO))
    p.add_argument("--contexto", default="Every effort moves you")
    p.add_argument("--tokens", type=int, default=50)
    p.add_argument("--temperatura", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--semente", type=int, default=123)
    a = p.parse_args()

    if not Path(a.checkpoint).exists():
        print(f"checkpoint não encontrado: {a.checkpoint}\n"
              "rode antes: uv run python experiments/sprint05-treinamento/treinar_base.py")
        return 1

    texto, dados = gerar_do_checkpoint(
        a.checkpoint, a.contexto, a.tokens, a.temperatura, a.top_k, a.semente
    )
    cfg, treino = dados["cfg_modelo"], dados["cfg_treino"] or {}
    print(f"checkpoint : {a.checkpoint}")
    print(f"modelo     : {dados['_parametros']:,} parâmetros | emb_dim={cfg['emb_dim']} "
          f"n_layers={cfg['n_layers']} n_heads={cfg['n_heads']} context_length={cfg['context_length']}")
    print(f"treino     : {treino.get('num_epocas')} épocas, lr={treino.get('taxa_aprendizado')}, "
          f"semente={treino.get('semente')} | gravado com torch {dados['versoes']['torch']}")
    print(f"geração    : temperatura={a.temperatura}, top_k={a.top_k}, {a.tokens} tokens")
    print(f"\n{texto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Salvamento e carregamento dos pesos do GPT — Sprint 5 (seção 5.4).

Um checkpoint é um único arquivo ``.pt`` (gravado com ``torch.save``) contendo
tudo o que é preciso para reconstruir o modelo e retomar o trabalho::

    {
      "modelo":       state_dict do GPTModel      (os pesos)
      "otimizador":   state_dict do otimizador    (momentos do AdamW; opcional)
      "cfg_modelo":   dicionário da arquitetura   (vocab_size, emb_dim, ...)
      "cfg_treino":   ConfigTreinamento.para_dict()  (opcional)
      "historico":    perdas e amostras do treino     (opcional)
      "versoes":      versões do Python e do PyTorch que gravaram o arquivo
    }

Guardar a configuração junto dos pesos é o que permite identificar o modelo
depois: o ``state_dict`` sozinho só tem tensores, e para recriar o
``GPTModel`` é preciso saber a arquitetura. O estado do otimizador é
necessário para *retomar* o treinamento — o AdamW mantém médias móveis dos
gradientes, e recomeçar sem elas muda a trajetória da otimização.

Os arquivos ``*.pt`` não são versionados no Git (ver ``.gitignore``).
"""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from gpt import GPTModel

__all__ = ["salvar_checkpoint", "carregar_checkpoint"]


def salvar_checkpoint(
    caminho: Path | str,
    modelo: GPTModel,
    otimizador: torch.optim.Optimizer | None = None,
    cfg_treino: dict | None = None,
    historico: dict | None = None,
) -> Path:
    """Grava pesos, configuração e (opcionalmente) otimizador e histórico."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict() if otimizador is not None else None,
        "cfg_modelo": dict(modelo.cfg),
        "cfg_treino": dict(cfg_treino) if cfg_treino is not None else None,
        "historico": historico,
        "versoes": {"python": platform.python_version(), "torch": str(torch.__version__)},
    }
    torch.save(dados, caminho)
    return caminho


def carregar_checkpoint(
    caminho: Path | str,
    dispositivo: torch.device | str = "cpu",
) -> tuple[GPTModel, dict]:
    """Reconstrói o ``GPTModel`` a partir de um checkpoint.

    O modelo é recriado com a ``cfg_modelo`` gravada, recebe os pesos e volta
    em modo ``eval()``, pronto para inferência. ``weights_only=True`` faz o
    ``torch.load`` aceitar apenas tensores e tipos básicos, sem executar código
    arbitrário guardado no arquivo.

    Returns:
        ``(modelo, dados)``, em que ``dados`` é o dicionário completo do arquivo.
    """
    dados = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelo = GPTModel(dados["cfg_modelo"]).to(dispositivo)
    modelo.load_state_dict(dados["modelo"])
    modelo.eval()
    return modelo, dados

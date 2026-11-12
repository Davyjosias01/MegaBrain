"""Treinamento e avaliação do GPT — Sprint 5 (Capítulo 5 do livro-texto).

Avaliação quantitativa do modelo pela perda de entropia cruzada (seção 5.1).

O modelo devolve, para cada posição da entrada, ``V`` logits — uma pontuação
para cada token do vocabulário ser o próximo. A tarefa é prever o token
seguinte, que é exatamente o alvo produzido pela janela deslizante da Sprint 2
(a entrada deslocada de uma posição).

A **entropia cruzada** mede o quanto essa previsão erra: para cada posição,
transforma os logits em probabilidades com softmax e toma ``-log`` da
probabilidade atribuída ao token correto. A perda do lote é a média desse valor
sobre todas as ``b · L`` posições::

    logits (b, L, V) -> flatten -> (b·L, V)
    alvos  (b, L)    -> flatten -> (b·L,)
    perda = média( -log softmax(logits)[alvo] )

Um modelo que acerta com certeza tem perda 0. Um modelo que distribui a
probabilidade uniformemente entre os ``V`` tokens tem perda ``ln(V)`` — para o
vocabulário do GPT-2, ``ln(50257) ≈ 10,82``. É perto disso que um modelo recém
inicializado começa.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "perda_do_lote",
    "perda_do_loader",
    "perplexidade",
    "avaliar_modelo",
    "treinar_modelo",
    "ConfigTreinamento",
]


def perda_do_lote(
    entrada: torch.Tensor,
    alvo: torch.Tensor,
    modelo: nn.Module,
    dispositivo: torch.device | str,
) -> torch.Tensor:
    """Perda de entropia cruzada de um lote — ``calc_loss_batch`` da seção 5.1.

    ``entrada`` e ``alvo`` têm forma ``(b, L)`` com token IDs. Devolve um tensor
    escalar ainda ligado ao grafo de computação, para que o laço de treinamento
    possa chamar ``backward()`` sobre ele.
    """
    entrada = entrada.to(dispositivo)
    alvo = alvo.to(dispositivo)
    logits = modelo(entrada)  # (b, L, V)
    return F.cross_entropy(logits.flatten(0, 1), alvo.flatten())


def perda_do_loader(
    loader,
    modelo: nn.Module,
    dispositivo: torch.device | str,
    num_lotes: int | None = None,
) -> float:
    """Perda média sobre os lotes de um DataLoader — ``calc_loss_loader`` da seção 5.1.

    Usada para avaliar o modelo nos conjuntos de treinamento e de validação.
    Roda em modo de avaliação (sem dropout) e sem calcular gradientes; o modo
    anterior do modelo é restaurado ao final.

    ``num_lotes`` limita a avaliação aos primeiros lotes, o que torna a
    avaliação periódica durante o treinamento mais barata. Se for maior que o
    número de lotes disponíveis, todos são usados. Um loader vazio devolve
    ``nan``, já que não há média a calcular.
    """
    total_lotes = len(loader)
    if total_lotes == 0:
        return float("nan")
    num_lotes = total_lotes if num_lotes is None else min(num_lotes, total_lotes)

    estava_treinando = modelo.training
    modelo.eval()
    soma = 0.0
    try:
        with torch.no_grad():
            for i, (entrada, alvo) in enumerate(loader):
                if i >= num_lotes:
                    break
                soma += perda_do_lote(entrada, alvo, modelo, dispositivo).item()
    finally:
        modelo.train(estava_treinando)
    return soma / num_lotes


def perplexidade(perda: float | torch.Tensor) -> float:
    """Perplexidade correspondente a uma perda de entropia cruzada — seção 5.1.

    ``perplexidade = exp(perda)``. É a mesma informação da perda, numa escala
    mais fácil de interpretar: o número efetivo de tokens entre os quais o
    modelo está "em dúvida" a cada previsão. Um modelo que espalha a
    probabilidade uniformemente pelo vocabulário tem perplexidade igual a ``V``
    (50.257 no GPT-2); um modelo perfeito tem perplexidade 1.
    """
    return math.exp(float(perda))


# --- Sprint 5: rotina de treinamento (seção 5.2) ------------------------------
def avaliar_modelo(
    modelo: nn.Module,
    loader_treino,
    loader_validacao,
    dispositivo: torch.device | str,
    num_lotes: int | None = None,
) -> tuple[float, float]:
    """Perdas de treinamento e de validação — ``evaluate_model`` da seção 5.2."""
    perda_treino = perda_do_loader(loader_treino, modelo, dispositivo, num_lotes)
    perda_validacao = perda_do_loader(loader_validacao, modelo, dispositivo, num_lotes)
    return perda_treino, perda_validacao


def treinar_modelo(
    modelo: nn.Module,
    loader_treino,
    loader_validacao,
    otimizador: torch.optim.Optimizer,
    dispositivo: torch.device | str,
    num_epocas: int,
    frequencia_avaliacao: int = 5,
    lotes_avaliacao: int | None = 5,
) -> dict[str, list]:
    """Laço de treinamento do GPT — ``train_model_simple`` da seção 5.2.

    Para cada época e cada lote de treinamento:

    1. ``otimizador.zero_grad()`` — zera os gradientes do passo anterior;
    2. ``perda_do_lote`` — passagem direta e entropia cruzada;
    3. ``perda.backward()`` — retropropagação: gradiente da perda em cada peso;
    4. ``otimizador.step()`` — atualiza os pesos na direção que reduz a perda.

    A cada ``frequencia_avaliacao`` passos, as perdas de treinamento e de
    validação são medidas em ``lotes_avaliacao`` lotes e registradas, junto com
    o número de tokens já vistos — é esse histórico que mostra se o modelo está
    aprendendo e quando começa o sobreajuste.

    Returns:
        ``{"passo", "epoca", "tokens_vistos", "perda_treino", "perda_validacao"}``,
        listas alinhadas, uma entrada por avaliação.
    """
    historico: dict[str, list] = {
        "passo": [],
        "epoca": [],
        "tokens_vistos": [],
        "perda_treino": [],
        "perda_validacao": [],
    }
    tokens_vistos = 0
    passo = -1

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        for entrada, alvo in loader_treino:
            otimizador.zero_grad()
            perda = perda_do_lote(entrada, alvo, modelo, dispositivo)
            perda.backward()
            otimizador.step()
            tokens_vistos += entrada.numel()
            passo += 1

            if passo % frequencia_avaliacao == 0:
                perda_treino, perda_validacao = avaliar_modelo(
                    modelo, loader_treino, loader_validacao, dispositivo, lotes_avaliacao
                )
                historico["passo"].append(passo)
                historico["epoca"].append(epoca)
                historico["tokens_vistos"].append(tokens_vistos)
                historico["perda_treino"].append(perda_treino)
                historico["perda_validacao"].append(perda_validacao)
                print(
                    f"  época {epoca:>3} | passo {passo:>5} | "
                    f"perda treino {perda_treino:.3f} | perda validação {perda_validacao:.3f}"
                )
    return historico


# --- Sprint 5: registro da configuração do treinamento -----------------------
@dataclass
class ConfigTreinamento:
    """Parâmetros de um treinamento, registrados para reprodução.

    Os valores padrão são os da seção 5.2 do livro. A configuração da
    arquitetura (``vocab_size``, ``emb_dim``…) fica no dicionário do modelo
    (``config_gpt2``); esta classe guarda o que é do *treinamento*.
    """

    num_epocas: int = 10
    batch_size: int = 2
    max_length: int = 256
    stride: int = 256
    proporcao_treino: float = 0.9
    otimizador: str = "AdamW"
    taxa_aprendizado: float = 4e-4
    weight_decay: float = 0.1
    frequencia_avaliacao: int = 5
    lotes_avaliacao: int = 5
    semente: int = 123
    dispositivo: str = "cpu"
    corpus: str = "sprints/the-verdict.txt"

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def de_dict(cls, dados: dict) -> "ConfigTreinamento":
        """Reconstrói a configuração, rejeitando campos desconhecidos."""
        conhecidos = {f.name for f in fields(cls)}
        desconhecidos = set(dados) - conhecidos
        if desconhecidos:
            raise ValueError(f"campos desconhecidos: {sorted(desconhecidos)}")
        return cls(**dados)

    def salvar_json(self, caminho: Path | str) -> None:
        Path(caminho).write_text(
            json.dumps(self.para_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def carregar_json(cls, caminho: Path | str) -> "ConfigTreinamento":
        return cls.de_dict(json.loads(Path(caminho).read_text(encoding="utf-8")))

    def criar_otimizador(self, modelo: nn.Module) -> torch.optim.Optimizer:
        """Instancia o otimizador registrado, com a taxa e o weight decay da configuração."""
        classe = getattr(torch.optim, self.otimizador)
        return classe(modelo.parameters(), lr=self.taxa_aprendizado, weight_decay=self.weight_decay)

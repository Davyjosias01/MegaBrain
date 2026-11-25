"""Demonstração do pipeline completo do projeto — integração final.

Percorre, na ordem em que foram construídas, todas as etapas do MegaBrain,
usando apenas os módulos de ``src/``:

====  ==========================================  ==========================
Etapa  O que acontece                              Módulo (sprint)
====  ==========================================  ==========================
1      texto -> token IDs (BPE)                    ``tokenizacao`` (2)
2      janela deslizante -> pares entrada/alvo     ``dados`` (2)
3      token IDs -> embeddings + posição           ``embeddings`` (2)
4      atenção causal com várias cabeças           ``attention`` (3)
5      blocos Transformer -> logits                ``gpt`` (4)
6      treinamento: entropia cruzada + AdamW       ``treinamento`` (5)
7      geração: greedy, temperatura e top-k        ``geracao`` (4 e 5)
8      pesos pré-treinados do GPT-2                ``pesos_pretreinados`` (6)
9      fine-tuning de classificação                ``classificacao`` (6)
10     fine-tuning de instruções                   ``instrucoes`` (6)
====  ==========================================  ==========================

Sem argumentos, tudo roda em CPU em cerca de um minuto, com modelos pequenos
treinados na hora: o objetivo é mostrar as peças encaixadas, não obter um bom
modelo. Com ``--completo``, as etapas 8 a 10 usam o GPT-2 124M e os
checkpoints ajustados que estiverem em ``modelos/`` (gerados pelos scripts de
``experiments/sprint06-fine-tuning/``); o que não existir é pulado com aviso.

Execução, a partir da raiz do repositório::

    uv run python src/demonstracao_pipeline.py
    uv run python src/demonstracao_pipeline.py --completo
"""

from __future__ import annotations

import contextlib
import gc
import io
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import MultiHeadAttention  # noqa: E402
from checkpoint import carregar_checkpoint  # noqa: E402
from classificacao import (  # noqa: E402
    PASTA_SPAM,
    DatasetDeClassificacao,
    acuracia_do_loader,
    adaptar_para_classificacao,
    carregar_classificador,
    classificar_texto,
    contar_treinaveis,
    treinar_classificador,
)
from dados import carregar_texto, criar_dataloader, criar_loaders_treino_validacao  # noqa: E402
from embeddings import EmbeddingDeEntrada  # noqa: E402
from geracao import gerar, ids_para_texto, texto_para_ids  # noqa: E402
from gpt import GPTModel, config_gpt2  # noqa: E402
from instrucoes import (  # noqa: E402
    DatasetDeInstrucoes,
    agrupar_lote,
    formatar_exemplo,
    montar_prompt,
    preparar_instrucoes,
    responder,
)
from pesos_pretreinados import PASTA_MODELOS, caminho_dos_pesos, carregar_gpt2_pretreinado  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402
from treinamento import perplexidade, treinar_modelo  # noqa: E402

SEMENTE = 123
CONTEXTO = "Every effort moves you"
CHECKPOINT_CLASSIFICADOR = PASTA_MODELOS / "classificador-spam.pt"
CHECKPOINT_INSTRUCOES = PASTA_MODELOS / "instrucoes-gpt2.pt"
MENSAGENS = [
    "You are a winner! Call now to claim your free prize of 1000 cash.",
    "Hey, are we still meeting for dinner tonight?",
]
TK = TokenizadorBPE()


def _etapa(numero: int, titulo: str) -> None:
    print(f"\n[{numero:>2}] {titulo}")
    print("     " + "-" * 66)


def _linha(rotulo: str, valor) -> None:
    print(f"     {rotulo:<28}: {valor}")


def _cfg_pequena(**alteracoes) -> dict:
    cfg = config_gpt2("small", emb_dim=128, n_heads=4, n_layers=2, context_length=64,
                      drop_rate=0.0, qkv_bias=True)
    cfg.update(alteracoes)
    return cfg


def etapas_de_construcao() -> GPTModel:
    """Etapas 1 a 7: do texto bruto ao modelo treinado que gera texto."""
    texto = carregar_texto()

    _etapa(1, "Texto -> token IDs (tokenização BPE)")
    frase = texto[:60]
    ids = TK.encode(frase)
    _linha("texto", repr(frase))
    _linha("token IDs", f"{ids[:10]} ... ({len(ids)} tokens)")
    _linha("tokens", [TK.decode([i]) for i in ids[:8]])
    _linha("ida e volta sem perda", TK.decode(ids) == frase)

    _etapa(2, "Janela deslizante -> pares (entrada, alvo)")
    loader = criar_dataloader(texto, batch_size=4, max_length=8, stride=8, shuffle=False)
    entrada, alvo = next(iter(loader))
    _linha("lote", f"entrada {tuple(entrada.shape)}, alvo {tuple(alvo.shape)}")
    _linha("entrada[0]", entrada[0].tolist())
    _linha("alvo[0] (deslocado em 1)", alvo[0].tolist())

    _etapa(3, "Token IDs -> embeddings de token + posição")
    torch.manual_seed(SEMENTE)
    camada = EmbeddingDeEntrada(vocab_size=50257, d_emb=128, context_length=8)
    vetores = camada(entrada)
    _linha("forma", f"{tuple(entrada.shape)} -> {tuple(vetores.shape)}")

    _etapa(4, "Atenção causal com várias cabeças")
    atencao = MultiHeadAttention(d_in=128, d_out=128, context_length=8, dropout=0.0, num_heads=4)
    contexto, pesos = atencao(vetores, return_attn_weights=True)
    futuro = torch.triu(torch.ones(8, 8, dtype=torch.bool), 1)
    _linha("vetores de contexto", tuple(contexto.shape))
    _linha("pesos de atenção", tuple(pesos.shape))
    _linha("peso sobre tokens futuros", f"{pesos[..., futuro].abs().max().item():.1f} (máscara causal)")
    _linha("cada linha soma", f"{pesos.sum(-1).mean().item():.3f}")

    _etapa(5, "Blocos Transformer -> logits (arquitetura GPT)")
    torch.manual_seed(SEMENTE)
    modelo = GPTModel(_cfg_pequena())
    with torch.no_grad():
        logits = modelo(entrada)
    _linha("configuração", "emb_dim 128, 2 blocos, 4 cabeças, contexto 64")
    _linha("parâmetros", f"{sum(p.numel() for p in modelo.parameters()):,}")
    _linha("logits", f"{tuple(logits.shape)}: um vetor de 50.257 notas por posição")

    _etapa(6, "Treinamento: entropia cruzada + AdamW")
    lt, lv = criar_loaders_treino_validacao(texto[:6000], batch_size=8, max_length=32)
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        h = treinar_modelo(modelo, lt, lv, torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=0.1),
                           "cpu", num_epocas=25, frequencia_avaliacao=20, lotes_avaliacao=2)
    _linha("corpus", f"{len(texto[:6000]):,} caracteres de the-verdict.txt, {len(lt)} lotes por época")
    _linha("perda de treino", f"{h['perda_treino'][0]:.2f} -> {h['perda_treino'][-1]:.2f} "
                              f"(ln 50257 = {math.log(50257):.2f})")
    _linha("perda de validação", f"{h['perda_validacao'][0]:.2f} -> {h['perda_validacao'][-1]:.2f} "
                                 f"(perplexidade {perplexidade(h['perda_validacao'][-1]):.0f})")

    _etapa(7, "Geração autorregressiva")
    modelo.eval()
    ids = texto_para_ids(CONTEXTO, TK)
    for nome, opcoes in [("greedy", {}), ("temperatura 1,0 + top-k 25", {"temperatura": 1.0, "top_k": 25})]:
        torch.manual_seed(SEMENTE)
        gerado = gerar(modelo, ids, 15, modelo.cfg["context_length"], **opcoes)
        _linha(nome, repr(ids_para_texto(gerado, TK)))
    print("     (modelo minúsculo, poucos passos: o texto mostra o mecanismo, não a qualidade)")
    return modelo


def etapa_pesos(completo: bool) -> None:
    _etapa(8, "Pesos pré-treinados do GPT-2 no mesmo GPTModel")
    if not completo:
        print("     pulada: use --completo para carregar o GPT-2 124M (arquivo de 548 MB).")
        return
    if not caminho_dos_pesos("small").exists():
        print("     arquivo de pesos ausente em modelos/: será baixado do Hugging Face Hub.")
    modelo = carregar_gpt2_pretreinado("small")
    ids = texto_para_ids(CONTEXTO, TK)
    _linha("parâmetros", f"{sum(p.numel() for p in modelo.parameters()):,}")
    _linha("greedy", repr(ids_para_texto(gerar(modelo, ids, 20, 1024), TK)))
    del modelo
    gc.collect()


def etapa_classificacao(completo: bool) -> None:
    _etapa(9, "Fine-tuning de classificação (spam × não spam)")
    if completo and CHECKPOINT_CLASSIFICADOR.exists():
        modelo, dados = carregar_classificador(CHECKPOINT_CLASSIFICADOR)
        max_length = dados["cfg_treino"]["max_length"]
        del dados
        _linha("modelo", "GPT-2 124M ajustado (modelos/classificador-spam.pt)")
    else:
        if completo:
            print("     checkpoint ausente: execute experiments/sprint06-fine-tuning/treinar_classificador.py.")
        max_length = 64
        treino = pd.read_csv(PASTA_SPAM / "treino.csv").head(240)
        validacao = pd.read_csv(PASTA_SPAM / "validacao.csv")
        lt = DataLoader(DatasetDeClassificacao(treino, TK, max_length=max_length), batch_size=8, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(SEMENTE))
        lv = DataLoader(DatasetDeClassificacao(validacao, TK, max_length=max_length), batch_size=8)
        torch.manual_seed(SEMENTE)
        modelo = adaptar_para_classificacao(GPTModel(_cfg_pequena()), num_classes=2, blocos_treinaveis=None)
        antes = acuracia_do_loader(lv, modelo, "cpu")
        treinar_classificador(modelo, lt, lv, torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=0.1),
                              "cpu", num_epocas=4, verboso=False)
        _linha("modelo", "GPT pequeno, sem pré-treino, com cabeça Linear(128, 2)")
        _linha("parâmetros treináveis", "{:,} de {:,}".format(*contar_treinaveis(modelo)))
        _linha("acurácia na validação", f"{antes:.1%} -> {acuracia_do_loader(lv, modelo, 'cpu'):.1%} "
                                        f"(240 mensagens, 4 épocas)")
    for mensagem in MENSAGENS:
        classe, probs = classificar_texto(mensagem, modelo, TK, max_length=max_length)
        _linha(f"-> {classe} ({probs[classe]:.0%})", repr(mensagem))
    del modelo
    gc.collect()


def etapa_instrucoes(completo: bool) -> None:
    _etapa(10, "Fine-tuning de instruções")
    treino, validacao, teste = preparar_instrucoes()
    exemplo = teste[0]
    ds = DatasetDeInstrucoes(treino[:8], TK)
    entradas, alvos = agrupar_lote([ds[i] for i in range(8)])
    _linha("exemplos", f"treino {len(treino)}, validação {len(validacao)}, teste {len(teste)}")
    _linha("lote", f"entrada {tuple(entradas.shape)}, alvo {tuple(alvos.shape)}, "
                   f"{int((alvos == -100).sum())} posições ignoradas na perda")
    print("     prompt de treino:")
    for linha in formatar_exemplo(exemplo).splitlines():
        print(f"       | {linha}")
    if completo and CHECKPOINT_INSTRUCOES.exists():
        modelo, dados = carregar_checkpoint(CHECKPOINT_INSTRUCOES, "cpu")
        del dados
        _linha("modelo", "GPT-2 124M ajustado (modelos/instrucoes-gpt2.pt)")
        _linha("resposta esperada", repr(exemplo["output"]))
        _linha("resposta gerada", repr(responder(exemplo, modelo, TK, max_novos_tokens=48)))
    else:
        if completo:
            print("     checkpoint ausente: execute experiments/sprint06-fine-tuning/treinar_instrucoes.py.")
        _linha("prompt de inferência", f"termina em {montar_prompt(exemplo)[-16:]!r}; o modelo continua daí")
        print("     geração da resposta com o modelo ajustado: use --completo.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    completo = "--completo" in sys.argv[1:]
    print("=" * 72)
    print("MegaBrain — pipeline completo: do texto bruto ao modelo ajustado")
    print("=" * 72)
    etapas_de_construcao()
    etapa_pesos(completo)
    etapa_classificacao(completo)
    etapa_instrucoes(completo)
    print("\n" + "=" * 72)
    print("Pipeline concluído: 10 etapas" + ("" if completo else " (8 a 10 em versão reduzida; veja --completo)"))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

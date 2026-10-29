"""Validação numérica da arquitetura GPT e da geração — Sprint 4.

Confere ``gpt.py`` e ``geracao.py`` contra os valores de referência do
Capítulo 4 (e da seção 5.3, para temperatura e top-k) e contra as propriedades
que cada componente deve satisfazer.

Os testes citam a seção do livro, e não a página, pela mesma razão das Sprints 2
e 3: o PDF em ``documentation/`` é uma edição preliminar.

Execução, a partir da raiz do repositório::

    python src/validar_gpt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geracao import (  # noqa: E402
    ajustar_logits,
    gerar,
    gerar_texto_simples,
    ids_para_texto,
    selecionar_proximo_token,
    texto_para_ids,
)
from gpt import (  # noqa: E402
    CONFIG_GPT2_124M,
    GELU,
    FeedForward,
    GPTModel,
    LayerNorm,
    RedeComAtalhos,
    TransformerBlock,
    config_gpt2,
    contar_parametros,
    estimar_parametros,
    parametros_por_componente,
    tamanho_em_mb,
)
from tokenizacao import TokenizadorBPE  # noqa: E402

TOLERANCIA = 1e-4
_resultados: list[tuple[bool, str]] = []


def _verificar(condicao: bool, descricao: str) -> None:
    _resultados.append((bool(condicao), descricao))
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}")


def _proximo(obtido: torch.Tensor, esperado: torch.Tensor, atol: float = TOLERANCIA) -> bool:
    return torch.allclose(obtido, esperado, atol=atol)


def _lote_do_livro() -> torch.Tensor:
    bpe = TokenizadorBPE()
    return torch.tensor([bpe.encode("Every effort moves you"), bpe.encode("Every day holds a")])


# ---------------------------------------------------------------------------
def teste_layernorm() -> None:
    """Seção 4.2 — layer normalization."""
    print("\n1. LayerNorm (seção 4.2)")

    torch.manual_seed(123)
    exemplo = torch.randn(2, 5)
    camada = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    saida = camada(exemplo)
    _verificar(
        _proximo(saida, torch.tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
                                      [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]])),
        "saída da camada de exemplo confere com o livro",
    )
    _verificar(
        _proximo(saida.mean(-1), torch.tensor([0.1324, 0.2170]))
        and _proximo(saida.var(-1), torch.tensor([0.0231, 0.0398])),
        "média e variância antes da normalização conferem",
    )

    ln = LayerNorm(emb_dim=6)
    normalizada = ln(saida)
    _verificar(
        _proximo(normalizada.mean(-1), torch.zeros(2))
        and _proximo(normalizada.var(-1, unbiased=False), torch.ones(2), atol=1e-3),
        "depois da LayerNorm cada linha tem média 0 e variância 1",
    )
    referencia = nn.LayerNorm(6, eps=1e-5)
    _verificar(_proximo(normalizada, referencia(saida)), "coincide com torch.nn.LayerNorm na inicialização")
    _verificar(
        contar_parametros(ln) == 12 and torch.equal(ln.scale.data, torch.ones(6)),
        "scale (1) e shift (0) são 2·emb_dim parâmetros treináveis",
    )


def teste_gelu_feedforward() -> None:
    """Seção 4.3 — GELU e feed forward."""
    print("\n2. GELU e FeedForward (seção 4.3)")

    x = torch.linspace(-3, 3, 101)
    gelu = GELU()(x)
    _verificar(
        _proximo(gelu, torch.nn.functional.gelu(x, approximate="tanh")),
        "GELU coincide com a aproximação tanh do PyTorch",
    )
    _verificar(
        bool((gelu[x < 0] < 0).any()) and bool((gelu[x < 0] > -0.2).all()),
        "para x < 0 a GELU é levemente negativa (a ReLU seria exatamente 0)",
    )

    ff = FeedForward(CONFIG_GPT2_124M)
    entrada = torch.rand(2, 3, 768)
    _verificar(ff(entrada).shape == (2, 3, 768), "FeedForward preserva a forma (2, 3, 768)")
    _verificar(ff.layers[0].out_features == 3072, "a camada interna expande 768 -> 3.072")
    _verificar(contar_parametros(ff) == 4_722_432, "FeedForward tem 4.722.432 parâmetros (exercício 4.1)")


def teste_atalhos() -> None:
    """Seção 4.4 — conexões residuais."""
    print("\n3. Conexões residuais (seção 4.4)")

    def medias_dos_gradientes(usar_atalhos: bool) -> list[float]:
        torch.manual_seed(123)
        rede = RedeComAtalhos([3, 3, 3, 3, 3, 1], usar_atalhos)
        saida = rede(torch.tensor([[1.0, 0.0, -1.0]]))
        nn.MSELoss()(saida, torch.tensor([[0.0]])).backward()
        return [p.grad.abs().mean().item() for n, p in rede.named_parameters() if "weight" in n]

    sem = medias_dos_gradientes(False)
    com = medias_dos_gradientes(True)
    _verificar(
        _proximo(torch.tensor(sem), torch.tensor([0.00020173587836325169, 0.0001201116101583466,
                                                  0.0007152041653171182, 0.001398873864673078,
                                                  0.005049646366387606]), atol=1e-7),
        "gradientes sem atalhos conferem com o livro",
    )
    _verificar(
        _proximo(torch.tensor(com), torch.tensor([0.22169792652130127, 0.20694105327129364,
                                                  0.32896995544433594, 0.2665732502937317,
                                                  1.3258541822433472]), atol=1e-5),
        "gradientes com atalhos conferem com o livro",
    )
    _verificar(com[0] / sem[0] > 1000, "na primeira camada o atalho aumenta o gradiente mais de 1.000 vezes")


def teste_bloco() -> None:
    """Seção 4.5 — bloco Transformer."""
    print("\n4. TransformerBlock (seção 4.5)")

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    bloco = TransformerBlock(CONFIG_GPT2_124M)
    _verificar(bloco(x).shape == (2, 4, 768), "o bloco preserva a forma (2, 4, 768)")
    _verificar(
        contar_parametros(bloco.att) == 2_360_064,
        "a atenção do bloco tem 2.360.064 parâmetros (exercício 4.1)",
    )
    _verificar(
        contar_parametros(bloco.ff) == 2 * contar_parametros(bloco.att) + 2304,
        "a feed forward tem cerca do dobro de parâmetros da atenção",
    )

    bloco.eval()
    perturbado = x.clone()
    perturbado[:, 3] += 10.0
    _verificar(
        torch.equal(bloco(x)[:, :3], bloco(perturbado)[:, :3]),
        "o bloco é causal: alterar o token 4 não muda as saídas dos tokens 1 a 3",
    )


def teste_modelo() -> None:
    """Seção 4.6 — modelo GPT."""
    print("\n5. GPTModel (seção 4.6)")

    lote = _lote_do_livro()
    _verificar(
        lote.tolist() == [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]],
        "lote de entrada confere com o livro",
    )

    torch.manual_seed(123)
    modelo = GPTModel(CONFIG_GPT2_124M)
    saida = modelo(lote)  # em modo de treino, como no livro
    _verificar(saida.shape == (2, 4, 50257), "logits têm forma (2, 4, 50257)")
    esperado = {
        (0, 0): [0.3613, 0.4222, -0.0711, 0.3483, 0.4661, -0.2838],
        (0, 3): [-1.0076, 0.3418, -0.1190, 0.7195, 0.4023, 0.0532],
        (1, 0): [-0.2564, 0.0900, 0.0335, 0.2659, 0.4454, -0.6806],
        (1, 3): [-0.1565, 0.3926, 0.3288, 1.2630, -0.1858, 0.0388],
    }
    confere = all(
        _proximo(torch.cat((saida[b, t, :3], saida[b, t, -3:])).detach(), torch.tensor(v))
        for (b, t), v in esperado.items()
    )
    _verificar(confere, "valores dos logits (semente 123) conferem com o livro")

    total = contar_parametros(modelo)
    _verificar(total == 163_009_536, "o modelo tem 163.009.536 parâmetros")
    _verificar(
        contar_parametros(modelo, sem_cabeca=True) == 124_412_160,
        "descontando a cabeça de saída (weight tying): 124.412.160",
    )
    _verificar(round(tamanho_em_mb(total), 2) == 621.83, "ocupa 621,83 MB em float32")
    _verificar(estimar_parametros(CONFIG_GPT2_124M) == total, "a fórmula de estimar_parametros coincide com o modelo real")

    partes = parametros_por_componente(modelo)
    _verificar(
        partes["feed forward (todos os blocos)"] == 12 * 4_722_432
        and partes["cabeça de saída"] == 50257 * 768,
        "a distribuição por componente soma o total",
    )

    xl = config_gpt2("xl")
    _verificar(
        estimar_parametros(xl) == 1_637_792_000
        and estimar_parametros(dict(xl, compartilhar_pesos=True)) == 1_557_380_800
        and round(tamanho_em_mb(estimar_parametros(xl)), 2) == 6247.68,
        "GPT-2 XL: 1.637.792.000 parâmetros, 1.557.380.800 com tying, 6.247,68 MB (exercício 4.2)",
    )

    torch.manual_seed(123)
    amarrado = GPTModel(dict(CONFIG_GPT2_124M, compartilhar_pesos=True))
    _verificar(
        contar_parametros(amarrado) == 124_412_160
        and amarrado.out_head.weight is amarrado.embedding.token_emb.weight,
        "com compartilhar_pesos=True, cabeça e embedding são a mesma matriz",
    )


def teste_geracao() -> None:
    """Seção 4.7 — geração greedy."""
    print("\n6. Geração de texto (seção 4.7)")

    ids = texto_para_ids("Hello, I am")
    _verificar(ids.tolist() == [[15496, 11, 314, 716]] and ids.shape == (1, 4), "contexto inicial (1, 4)")

    torch.manual_seed(123)
    modelo = GPTModel(CONFIG_GPT2_124M).eval()
    saida = gerar_texto_simples(modelo, ids, max_novos_tokens=6, tamanho_contexto=1024)
    _verificar(
        saida.tolist() == [[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]],
        "os 6 tokens gerados conferem com o livro",
    )
    _verificar(
        ids_para_texto(saida) == "Hello, I am Featureiman Byeswickattribute argue",
        "texto gerado: 'Hello, I am Featureiman Byeswickattribute argue'",
    )
    _verificar(
        torch.equal(gerar(modelo, ids, 6, 1024, temperatura=0.0), saida),
        "temperatura 0 é a escolha greedy, determinística",
    )

    curto = gerar(modelo, ids, 3, tamanho_contexto=2)
    manual = ids
    for _ in range(3):
        with torch.no_grad():
            proximo = modelo(manual[:, -2:])[:, -1, :].argmax(-1, keepdim=True)
        manual = torch.cat((manual, proximo), dim=1)
    _verificar(torch.equal(curto, manual), "o contexto é truncado aos últimos tamanho_contexto tokens")

    primeiro = int(saida[0, 4])
    _verificar(
        gerar(modelo, ids, 6, 1024, id_fim=primeiro).shape == (1, 4),
        "a geração para ao produzir id_fim",
    )


def teste_selecao() -> None:
    """Seção 5.3 — temperatura e top-k."""
    print("\n7. Seleção do próximo token: temperatura e top-k (seção 5.3)")

    logits = torch.tensor([[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]])
    _verificar(selecionar_proximo_token(logits).item() == 3, "greedy escolhe 'forward' (ID 3)")

    torch.manual_seed(123)
    amostras = [selecionar_proximo_token(logits, temperatura=1.0).item() for _ in range(1000)]
    contagem = torch.bincount(torch.tensor(amostras), minlength=9).tolist()
    _verificar(
        contagem == [73, 0, 0, 582, 2, 0, 0, 343, 0],
        "1.000 sorteios com temperatura 1 conferem: 73 closer, 582 forward, 343 toward",
    )

    p = lambda t: torch.softmax(ajustar_logits(logits, t), dim=-1)  # noqa: E731
    _verificar(
        p(0.1).max() > 0.99 and p(5.0).max() < 0.35,
        "temperatura 0,1 concentra a distribuição; temperatura 5 a espalha",
    )

    filtrados = ajustar_logits(logits, temperatura=1.0, top_k=3)
    _verificar(
        _proximo(torch.softmax(filtrados, dim=-1),
                 torch.tensor([[0.0615, 0, 0, 0.5775, 0, 0, 0, 0.3610, 0]])),
        "top-k = 3 mantém só 'closer', 'forward' e 'toward', com 0,0615 / 0,5775 / 0,3610",
    )
    torch.manual_seed(1)
    sorteados = {selecionar_proximo_token(logits, temperatura=5.0, top_k=3).item() for _ in range(500)}
    _verificar(sorteados <= {0, 3, 7}, "com top-k, nem uma temperatura alta escolhe tokens fora dos k")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Validação da arquitetura GPT e da geração — Sprint 4")
    print("=" * 72)

    teste_layernorm()
    teste_gelu_feedforward()
    teste_atalhos()
    teste_bloco()
    teste_modelo()
    teste_geracao()
    teste_selecao()

    aprovados = sum(ok for ok, _ in _resultados)
    print("\n" + "=" * 72)
    print(f"{aprovados}/{len(_resultados)} verificações aprovadas")
    print("=" * 72)
    return 0 if aprovados == len(_resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())

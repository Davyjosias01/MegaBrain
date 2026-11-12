"""Validação do treinamento, da avaliação e dos checkpoints — Sprint 5.

Confere ``treinamento.py``, ``checkpoint.py`` e as funções de divisão de dados
de ``dados.py`` contra os valores de referência do Capítulo 5 e contra as
propriedades que cada etapa deve satisfazer. Reúne, num só lugar, os testes
usados em cada task da sprint (A1–A4, B1–B3, C1–C2).

Todos os modelos são pequenos e rodam em CPU em poucos segundos.

Execução, a partir da raiz do repositório::

    uv run python src/validar_treinamento.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import carregar_checkpoint, retomar_treinamento, salvar_checkpoint  # noqa: E402
from dados import (  # noqa: E402
    carregar_texto,
    criar_loaders_treino_validacao,
    dividir_treino_validacao,
    numero_de_amostras,
)
from gpt import GPTModel, config_gpt2  # noqa: E402
from tokenizacao import TokenizadorBPE  # noqa: E402
from treinamento import (  # noqa: E402
    ConfigTreinamento,
    gerar_amostra,
    perda_do_loader,
    perda_do_lote,
    perplexidade,
    treinar_modelo,
)

V = 50257
CONTEXTO = "Every effort moves you"
_resultados: list[tuple[bool, str]] = []


def _verificar(condicao: bool, descricao: str) -> None:
    _resultados.append((bool(condicao), descricao))
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}")


def _modelo_pequeno(drop_rate: float = 0.0, context_length: int = 32) -> GPTModel:
    torch.manual_seed(123)
    cfg = config_gpt2(
        "small", context_length=context_length, emb_dim=64, n_heads=4, n_layers=2,
        drop_rate=drop_rate,
    )
    return GPTModel(cfg)


def _lotes(num: int = 3, tamanho: int = 16) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randint(0, V, (2 * num, tamanho))
    y = torch.randint(0, V, (2 * num, tamanho))
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def teste_perda() -> None:
    """Seção 5.1 — entropia cruzada e perplexidade (A1–A3)."""
    print("\nPerda e perplexidade (seção 5.1)")
    m = _modelo_pequeno().eval()
    ld = _lotes()
    x, y = next(iter(ld))
    with torch.no_grad():
        p = perda_do_lote(x, y, m, "cpu")
        manual = -torch.log_softmax(m(x), -1).gather(-1, y.unsqueeze(-1)).mean()
    _verificar(torch.allclose(p, manual, atol=1e-5), "perda_do_lote = média de -log p(alvo)")
    _verificar(abs(p.item() - math.log(V)) < 1.0, f"modelo sem treino: perda {p.item():.2f} ≈ ln(V) = 10,82")

    with torch.no_grad():
        lotes = [perda_do_lote(a, b, m, "cpu").item() for a, b in ld]
    m.train()
    _verificar(math.isclose(perda_do_loader(ld, m, "cpu"), sum(lotes) / 3, abs_tol=1e-6),
               "perda_do_loader = média das perdas dos lotes")
    _verificar(math.isclose(perda_do_loader(ld, m, "cpu", 2), sum(lotes[:2]) / 2, abs_tol=1e-6),
               "num_lotes limita a avaliação aos primeiros lotes")
    _verificar(m.training, "perda_do_loader restaura o modo de treino")
    _verificar(all(q.grad is None for q in m.parameters()), "perda_do_loader não calcula gradientes")
    vazio = DataLoader(TensorDataset(torch.zeros(0, 4, dtype=torch.long)) , batch_size=2)
    _verificar(math.isnan(perda_do_loader(vazio, m, "cpu")), "loader vazio devolve nan")

    _verificar(math.isclose(perplexidade(math.log(V)), V, rel_tol=1e-9), "perplexidade(ln V) = V")
    _verificar(perplexidade(0.0) == 1.0, "perplexidade(0) = 1 (modelo perfeito)")


def teste_divisao() -> None:
    """Seção 5.1 — conjuntos de treinamento e validação (A4)."""
    print("\nDivisão treino/validação (seção 5.1)")
    texto = carregar_texto()
    treino, validacao = dividir_treino_validacao(texto, 0.9)
    _verificar(treino + validacao == texto, "a divisão não perde nem repete caracteres")
    _verificar(len(treino) == int(0.9 * len(texto)), "90% dos caracteres no treinamento")

    tok = TokenizadorBPE()
    lt, lv = criar_loaders_treino_validacao(texto, batch_size=2, max_length=256, tokenizador=tok)
    at = numero_de_amostras(len(tok.encode(treino)), 256, 256)
    av = numero_de_amostras(len(tok.encode(validacao)), 256, 256)
    _verificar(len(lt) == at // 2 == 9, f"treinamento: {len(lt)} lotes (livro: 9)")
    _verificar(len(lv) == math.ceil(av / 2) == 1, f"validação: {len(lv)} lote")
    x, y = next(iter(lt))
    _verificar(x.shape == (2, 256) and bool((x[:, 1:] == y[:, :-1]).all()),
               "lotes (2, 256) com alvo = entrada deslocada")
    try:
        dividir_treino_validacao(texto, 1.0)
        _verificar(False, "proporção inválida é rejeitada")
    except ValueError:
        _verificar(True, "proporção inválida é rejeitada")


def teste_treinamento() -> None:
    """Seção 5.2 — laço de treinamento, configuração e amostras (B1–B3)."""
    print("\nTreinamento (seção 5.2)")
    m = _modelo_pequeno()
    torch.manual_seed(1)
    x = torch.randint(0, V, (2, 32))
    y = torch.randint(0, V, (2, 32))
    ld = DataLoader(TensorDataset(x, y), batch_size=2)
    antes = perda_do_loader(ld, m, "cpu")
    pesos = m.out_head.weight.detach().clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    h = treinar_modelo(m, ld, ld, opt, "cpu", num_epocas=40, frequencia_avaliacao=10,
                       lotes_avaliacao=1, contexto_inicial=CONTEXTO, tokens_amostra=5)
    depois = perda_do_loader(ld, m, "cpu")
    _verificar(depois < 0.5 * antes, f"perda num lote fixo cai mais de 50% ({antes:.2f} -> {depois:.2f})")
    _verificar(not torch.equal(pesos, m.out_head.weight), "otimizador atualiza os pesos")
    _verificar(h["passo"] == [0, 10, 20, 30], "avaliação a cada frequencia_avaliacao passos")
    _verificar(h["tokens_vistos"][-1] == 31 * 64, "tokens vistos contados corretamente")
    _verificar(len(h["amostras"]) == 40 and h["amostras"][0].startswith(CONTEXTO),
               "uma amostra por época, começando pelo contexto")

    c = ConfigTreinamento(num_epocas=3, taxa_aprendizado=1e-3, semente=7)
    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "cfg.json"
        c.salvar_json(caminho)
        _verificar(ConfigTreinamento.carregar_json(caminho) == c, "configuração: ida e volta pelo JSON")
    o = c.criar_otimizador(m)
    _verificar(isinstance(o, torch.optim.AdamW) and o.param_groups[0]["lr"] == 1e-3,
               "configuração cria o otimizador registrado")

    m2 = _modelo_pequeno(drop_rate=0.1).train()
    a = gerar_amostra(m2, CONTEXTO, "cpu", max_novos_tokens=8)
    _verificar(m2.training and gerar_amostra(m2, CONTEXTO, "cpu", max_novos_tokens=8) == a,
               "amostra determinística e sem alterar o modo do modelo")


def teste_checkpoint() -> None:
    """Seção 5.4 — salvar, carregar e retomar (C1–C2)."""
    print("\nCheckpoints (seção 5.4)")
    ct = ConfigTreinamento(taxa_aprendizado=1e-3)
    ld = _lotes(num=4, tamanho=32)
    seq = list(ld) * 2

    def passos(modelo, otimizador, lotes):
        modelo.train()
        for a, b in lotes:
            otimizador.zero_grad()
            perda_do_lote(a, b, modelo, "cpu").backward()
            otimizador.step()

    direto = _modelo_pequeno()
    passos(direto, ct.criar_otimizador(direto), seq)

    m = _modelo_pequeno()
    opt = ct.criar_otimizador(m)
    passos(m, opt, seq[:3])
    m.eval()
    x = seq[0][0]
    with torch.no_grad():
        logits = m(x)
    texto = gerar_amostra(m, CONTEXTO, "cpu", max_novos_tokens=10)

    with tempfile.TemporaryDirectory() as d:
        caminho = salvar_checkpoint(Path(d) / "modelo.pt", m, opt, ct.para_dict(), {"amostras": [texto]})
        carregado, dados = carregar_checkpoint(caminho)
        with torch.no_grad():
            _verificar(torch.equal(carregado(x), logits), "modelo carregado: logits idênticos")
        _verificar(gerar_amostra(carregado, CONTEXTO, "cpu", max_novos_tokens=10) == texto,
                   "modelo carregado: mesma geração de texto")
        _verificar(dados["cfg_modelo"] == m.cfg and dados["cfg_treino"] == ct.para_dict(),
                   "configurações do modelo e do treino preservadas")
        retomado, opt2, _ = retomar_treinamento(caminho)
        passos(retomado, opt2, seq[3:])

    dif = max((a - b).abs().max().item() for a, b in zip(direto.parameters(), retomado.parameters()))
    _verificar(dif < 1e-5, f"treino retomado = treino sem interrupção (diferença {dif:.1e})")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Validação do treinamento e da avaliação — Sprint 5")
    print("=" * 72)

    teste_perda()
    teste_divisao()
    teste_treinamento()
    teste_checkpoint()

    aprovados = sum(ok for ok, _ in _resultados)
    print("\n" + "=" * 72)
    print(f"{aprovados}/{len(_resultados)} verificações aprovadas")
    print("=" * 72)
    return 0 if aprovados == len(_resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())

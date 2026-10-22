"""Validação numérica do pipeline de dados da Sprint 2.

Confere ``tokenizacao.py``, ``dados.py`` e ``embeddings.py`` contra os valores
de referência do Capítulo 2 do livro-texto e contra as propriedades que cada
etapa deve satisfazer.

Os testes citam a **seção** do livro, e não a página: o PDF em
``documentation/`` é uma edição preliminar, com numeração de páginas diferente
da edição final. Há também uma diferença de conteúdo, verificada no teste 1: a
edição preliminar usa uma expressão regular sem ``:`` e ``;`` e por isso obtém
4.649 tokens e vocabulário de 1.159; a edição final (e este projeto) separa
também esses dois sinais e obtém 4.690 tokens e vocabulário de 1.130. Os
valores do BPE, do DataLoader e dos embeddings são iguais nas duas edições.

Execução, a partir da raiz do repositório::

    python src/validar_dados.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import (  # noqa: E402
    GPTDatasetV1,
    carregar_texto,
    criar_dataloader,
    numero_de_amostras,
    pares_entrada_alvo,
)
from embeddings import EmbeddingDeEntrada  # noqa: E402
from tokenizacao import (  # noqa: E402
    DESCONHECIDO,
    FIM_DE_TEXTO,
    TokenizadorBPE,
    TokenizadorSimples,
    construir_vocabulario,
    dividir_em_tokens,
)

TOLERANCIA = 1e-4
TEXTO = carregar_texto()
FRASE_GISBURN = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
FRASES_CONCATENADAS = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

_resultados: list[tuple[bool, str]] = []


def _verificar(condicao: bool, descricao: str) -> None:
    _resultados.append((bool(condicao), descricao))
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}")


# ---------------------------------------------------------------------------
def teste_tokenizacao() -> None:
    """Seção 2.2 — dividir o texto em tokens."""
    print("\n1. Tokenização por expressão regular (seção 2.2)")

    _verificar(len(TEXTO) == 20479, "o corpus the-verdict.txt tem 20.479 caracteres")
    _verificar(
        dividir_em_tokens("Hello, world. Is this-- a test?")
        == ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"],
        "pontuação e '--' viram tokens; espaços são descartados",
    )

    tokens = dividir_em_tokens(TEXTO)
    _verificar(len(tokens) == 4690, "o corpus produz 4.690 tokens (edição final do livro)")
    _verificar(tokens[:4] == ["I", "HAD", "always", "thought"], "os primeiros tokens conferem")

    regex_preliminar = r'([,.?_!"()\']|--|\s)'
    tokens_preliminar = [t.strip() for t in re.split(regex_preliminar, TEXTO) if t.strip()]
    _verificar(
        (len(tokens_preliminar), len(set(tokens_preliminar))) == (4649, 1159),
        "a regex da edição preliminar (sem ':' e ';') reproduz os 4.649 / 1.159 do PDF",
    )


def teste_vocabulario() -> None:
    """Seções 2.3 e 2.4 — vocabulário, token IDs e tokens especiais."""
    print("\n2. Vocabulário e Token IDs (seções 2.3 e 2.4)")

    tokens = dividir_em_tokens(TEXTO)
    vocab_v1 = construir_vocabulario(tokens, tokens_especiais=())
    _verificar(len(vocab_v1) == 1130, "vocabulário sem tokens especiais tem 1.130 entradas")
    _verificar(
        list(vocab_v1.items())[:3] == [("!", 0), ('"', 1), ("'", 2)],
        "o vocabulário é ordenado: pontuação antes das letras",
    )

    v1 = TokenizadorSimples(vocab_v1)
    ids = v1.encode(FRASE_GISBURN)
    _verificar(
        ids == [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851,
                1108, 754, 793, 7],
        "encode da frase de Gisburn confere com a listagem 2.3",
    )
    _verificar(v1.encode(v1.decode(ids)) == ids, "decode seguido de encode recupera os mesmos IDs")
    _verificar(
        v1.ids_para_tokens(v1.tokens_para_ids(["the", "last", "pride"])) == ["the", "last", "pride"],
        "token -> ID -> token é uma bijeção sobre o vocabulário",
    )

    try:
        v1.encode("Hello, do you like tea?")
        _verificar(False, "palavra desconhecida sem <|unk|> gera erro")
    except KeyError:
        _verificar(True, "palavra desconhecida sem <|unk|> gera erro (comportamento da V1)")

    v2 = TokenizadorSimples.a_partir_do_texto(TEXTO)
    _verificar(v2.tamanho_vocabulario == 1132, "com <|endoftext|> e <|unk|> o vocabulário tem 1.132")
    _verificar(
        (v2.str_para_id[FIM_DE_TEXTO], v2.str_para_id[DESCONHECIDO]) == (1130, 1131),
        "os tokens especiais ocupam os IDs 1130 e 1131",
    )
    ids2 = v2.encode(FRASES_CONCATENADAS)
    _verificar(
        ids2 == [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7],
        "encode com <|unk|> e <|endoftext|> confere com a listagem 2.4",
    )
    _verificar(
        v2.decode(ids2)
        == "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.",
        "decode mostra a perda de informação: 'Hello' e 'palace' viram <|unk|>",
    )


def teste_bpe() -> None:
    """Seção 2.5 — Byte Pair Encoding do GPT-2."""
    print("\n3. Byte Pair Encoding (seção 2.5)")

    bpe = TokenizadorBPE()
    _verificar(bpe.tamanho_vocabulario == 50257, "o vocabulário do GPT-2 tem 50.257 tokens")
    _verificar(bpe.encode(FIM_DE_TEXTO) == [50256], "<|endoftext|> é o último ID, 50256")

    texto = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    ids = bpe.encode(texto)
    _verificar(
        ids == [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250,
                8812, 2114, 286, 617, 34680, 27271, 13],
        "IDs da frase de exemplo conferem com a seção 2.5",
    )
    _verificar(bpe.decode(ids) == texto, "o decode do BPE reconstrói o texto exatamente")

    desconhecida = bpe.encode("Akwirw ier")
    _verificar(
        desconhecida == [33901, 86, 343, 86, 220, 959]
        and bpe.decode(desconhecida) == "Akwirw ier",
        "palavra inexistente é quebrada em subpalavras, sem <|unk|> (exercício 2.1)",
    )
    _verificar(len(bpe.encode(TEXTO)) == 5145, "o corpus produz 5.145 tokens BPE")


def teste_pares() -> None:
    """Seção 2.6 — pares de entrada e alvo."""
    print("\n4. Pares entrada/alvo (seção 2.6)")

    amostra = TokenizadorBPE().encode(TEXTO)[50:]
    x, y = amostra[:4], amostra[1:5]
    _verificar(x == [290, 4920, 2241, 287] and y == [4920, 2241, 287, 257], "x e y conferem com o livro")
    _verificar(
        pares_entrada_alvo(amostra, 4)
        == [([290], 4920), ([290, 4920], 2241), ([290, 4920, 2241], 287),
            ([290, 4920, 2241, 287], 257)],
        "uma janela de 4 tokens contém 4 tarefas de previsão",
    )


def teste_dataloader() -> None:
    """Seção 2.6 — Dataset e DataLoader."""
    print("\n5. Dataset e DataLoader (seção 2.6)")

    carregador = criar_dataloader(TEXTO, batch_size=1, max_length=4, stride=1)
    lotes = iter(carregador)
    x1, y1 = next(lotes)
    x2, y2 = next(lotes)
    _verificar(
        x1.tolist() == [[40, 367, 2885, 1464]] and y1.tolist() == [[367, 2885, 1464, 1807]],
        "primeiro lote (batch 1, stride 1) confere com o livro",
    )
    _verificar(
        x2.tolist() == [[367, 2885, 1464, 1807]] and y2.tolist() == [[2885, 1464, 1807, 3619]],
        "com stride 1 o segundo lote é o primeiro deslocado de um token",
    )

    entradas, alvos = next(iter(criar_dataloader(TEXTO, batch_size=8, max_length=4, stride=4)))
    _verificar(entradas.shape == (8, 4) and alvos.shape == (8, 4), "lote (8, 4) para batch 8 e contexto 4")
    _verificar(
        entradas[:, 0].tolist() == [40, 1807, 10899, 15632, 922, 568, 1049, 284]
        and alvos[:, 0].tolist() == [367, 3619, 2138, 438, 5891, 340, 5975, 3285],
        "batch 8 / stride 4 confere com o livro",
    )
    _verificar(torch.equal(entradas[:, 1:], alvos[:, :-1]), "o alvo é a entrada deslocada de uma posição")
    bpe = TokenizadorBPE()
    ids_corpus = bpe.encode(TEXTO)
    _verificar(
        entradas.flatten().tolist() == ids_corpus[:32],
        "com stride = contexto as janelas são encostadas, sem sobreposição",
    )

    n = len(ids_corpus)
    configuracoes = [(4, 1), (4, 4), (256, 128), (256, 256), (1024, 512)]
    _verificar(
        all(
            numero_de_amostras(n, L, s) == len(GPTDatasetV1(TEXTO, bpe, L, s))
            for L, s in configuracoes
        ),
        "numero_de_amostras coincide com o tamanho real do dataset",
    )
    _verificar(
        numero_de_amostras(n, 256, 256) == 20
        and len(criar_dataloader(TEXTO, batch_size=8, max_length=256, stride=256)) == 2,
        "drop_last descarta o lote incompleto: 20 amostras -> 2 lotes de 8",
    )


def teste_embeddings() -> None:
    """Seções 2.7 e 2.8 — embeddings de token e posicionais."""
    print("\n6. Embeddings (seções 2.7 e 2.8)")

    torch.manual_seed(123)
    camada = nn.Embedding(6, 3)
    esperado = torch.tensor(
        [
            [0.3374, -0.1778, -0.1690],
            [0.9178, 1.5810, 1.3010],
            [1.2753, -0.2010, -0.1606],
            [-0.4015, 0.9666, -1.1481],
            [-1.1589, 0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096],
        ]
    )
    _verificar(torch.allclose(camada.weight, esperado, atol=TOLERANCIA), "pesos de nn.Embedding(6, 3) conferem (seed 123)")

    ids = torch.tensor([2, 3, 5, 1])
    _verificar(torch.equal(camada(ids), camada.weight[ids]), "o embedding é uma consulta de linhas da matriz")
    one_hot = torch.nn.functional.one_hot(ids, num_classes=6).float()
    _verificar(
        torch.allclose(one_hot @ camada.weight, camada(ids)),
        "equivale a one-hot @ matriz de pesos, sem materializar o one-hot",
    )

    torch.manual_seed(123)
    entrada = EmbeddingDeEntrada(vocab_size=50257, d_emb=256, context_length=4)
    lote, _ = next(iter(criar_dataloader(TEXTO, batch_size=8, max_length=4, stride=4)))
    saida = entrada(lote)
    _verificar(saida.shape == (8, 4, 256), "embeddings de entrada têm forma (8, 4, 256), como no livro")
    _verificar(
        sum(p.numel() for p in entrada.parameters()) == (50257 + 4) * 256,
        "parâmetros = (vocab_size + context_length) · d_emb = 12.866.816",
    )

    repetido = torch.tensor([[262, 262, 262]])  # " the" três vezes
    sem_posicao = entrada.token_emb(repetido)[0]
    com_posicao = entrada(repetido)[0]
    _verificar(
        torch.equal(sem_posicao[0], sem_posicao[2]),
        "sem posição, o mesmo token tem o mesmo vetor em qualquer lugar",
    )
    _verificar(
        not torch.allclose(com_posicao[0], com_posicao[2]),
        "com posição, o mesmo token passa a ter vetores diferentes",
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Validação do pipeline de dados — Sprint 2")
    print("=" * 72)

    teste_tokenizacao()
    teste_vocabulario()
    teste_bpe()
    teste_pares()
    teste_dataloader()
    teste_embeddings()

    aprovados = sum(ok for ok, _ in _resultados)
    print("\n" + "=" * 72)
    print(f"{aprovados}/{len(_resultados)} verificações aprovadas")
    print("=" * 72)
    return 0 if aprovados == len(_resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())

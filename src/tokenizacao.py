"""Tokenização e vocabulário — Sprint 2 (Capítulo 2 do livro-texto).

Primeira etapa do pipeline: transformar texto em uma sequência de inteiros que
uma rede neural possa consumir. O módulo implementa os dois caminhos estudados
no Capítulo 2 de RASCHKA, S. *Build a Large Language Model (From Scratch)*:

1. ``TokenizadorSimples`` — tokenização por expressão regular e vocabulário
   construído a partir do próprio corpus (seções 2.2 a 2.4). Didático: deixa
   explícita a relação ``token <-> token ID <-> vocabulário``.
2. ``TokenizadorBPE``     — *Byte Pair Encoding* do GPT-2, via ``tiktoken``
   (seção 2.5). É o tokenizador usado no restante do projeto.

Os dois expõem a mesma interface (``encode``, ``decode``,
``tamanho_vocabulario``, ``tokens_para_ids``, ``ids_para_tokens``), de modo
que os experimentos podem trocar um pelo outro sem alterar o código ao redor.

Fluxo::

    texto -> dividir_em_tokens -> tokens -> vocabulário -> token IDs
                                                       <- decode <-
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import tiktoken

__all__ = [
    "PADRAO_TOKENIZACAO",
    "FIM_DE_TEXTO",
    "DESCONHECIDO",
    "TOKENS_ESPECIAIS",
    "dividir_em_tokens",
    "construir_vocabulario",
    "TokenizadorSimples",
    "TokenizadorBPE",
]

# Separa em espaços em branco, nos sinais de pontuação e no travessão duplo
# ``--``. O grupo de captura ``( )`` faz o ``re.split`` devolver também os
# separadores, para que a pontuação vire token em vez de ser descartada.
PADRAO_TOKENIZACAO = r'([,.:;?_!"()\']|--|\s)'

# Tokens especiais (seção 2.4). Não aparecem no texto: são inseridos pelo
# próprio pipeline para sinalizar fronteiras e lacunas do vocabulário.
FIM_DE_TEXTO = "<|endoftext|>"  # separa documentos independentes concatenados
DESCONHECIDO = "<|unk|>"  # substitui palavras ausentes do vocabulário
TOKENS_ESPECIAIS = (FIM_DE_TEXTO, DESCONHECIDO)

# Na reconstrução do texto, a pontuação volta a "colar" na palavra anterior.
_ESPACO_ANTES_DA_PONTUACAO = re.compile(r'\s+([,.:;?!"()\'])')


def dividir_em_tokens(texto: str) -> list[str]:
    """Divide o texto em palavras e sinais de pontuação.

    Os espaços são usados como separadores e descartados: o modelo didático do
    capítulo não precisa deles, e removê-los encurta a sequência. O custo é que
    a formatação original (quebras de linha, espaços duplos) não pode ser
    recuperada no ``decode`` — limitação que o BPE não tem.

    >>> dividir_em_tokens("Hello, world. Is this-- a test?")
    ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
    """
    return [peca.strip() for peca in re.split(PADRAO_TOKENIZACAO, texto) if peca.strip()]


def construir_vocabulario(
    tokens: Iterable[str],
    tokens_especiais: Iterable[str] = TOKENS_ESPECIAIS,
) -> dict[str, int]:
    """Associa cada token distinto a um inteiro único.

    Os tokens são deduplicados e ordenados alfabeticamente, e os especiais são
    acrescentados ao final. A ordem não tem significado para o modelo — o ID é
    apenas um índice, que mais adiante seleciona uma linha da matriz de
    embedding —, mas fixá-la torna o vocabulário reprodutível.

    Args:
        tokens: sequência de tokens, tipicamente ``dividir_em_tokens(corpus)``.
        tokens_especiais: acrescentados ao final, se ainda não estiverem presentes.

    Returns:
        Dicionário ``token -> token ID``.
    """
    distintos = sorted(set(tokens))
    distintos += [t for t in tokens_especiais if t not in distintos]
    return {token: indice for indice, token in enumerate(distintos)}


class TokenizadorSimples:
    """Tokenizador por expressão regular com vocabulário fechado.

    Reúne as duas versões do livro (listagens 2.3 e 2.4) em uma única classe:

    - se o vocabulário contém ``<|unk|>``, palavras desconhecidas são mapeadas
      para ele (comportamento da ``SimpleTokenizerV2``);
    - se não contém, uma palavra desconhecida gera ``KeyError`` com mensagem
      explicativa (comportamento da ``SimpleTokenizerV1``).

    O vocabulário é *fechado*: só conhece as palavras do corpus de onde foi
    construído. É exatamente essa limitação que motiva o BPE.

    Args:
        vocabulario: dicionário ``token -> token ID``, como o devolvido por
            ``construir_vocabulario``.
    """

    def __init__(self, vocabulario: dict[str, int]) -> None:
        self.str_para_id = dict(vocabulario)
        self.id_para_str = {indice: token for token, indice in vocabulario.items()}

    @classmethod
    def a_partir_do_texto(
        cls, texto: str, tokens_especiais: Iterable[str] = TOKENS_ESPECIAIS
    ) -> "TokenizadorSimples":
        """Constrói o vocabulário a partir de um corpus e devolve o tokenizador."""
        return cls(construir_vocabulario(dividir_em_tokens(texto), tokens_especiais))

    @property
    def tamanho_vocabulario(self) -> int:
        return len(self.str_para_id)

    def tokens_para_ids(self, tokens: Iterable[str]) -> list[int]:
        """Converte tokens em IDs, aplicando ``<|unk|>`` quando disponível."""
        ids = []
        for token in tokens:
            if token in self.str_para_id:
                ids.append(self.str_para_id[token])
            elif DESCONHECIDO in self.str_para_id:
                ids.append(self.str_para_id[DESCONHECIDO])
            else:
                raise KeyError(
                    f"token {token!r} não está no vocabulário e o vocabulário não "
                    f"possui {DESCONHECIDO} para representá-lo."
                )
        return ids

    def ids_para_tokens(self, ids: Iterable[int]) -> list[str]:
        """Operação inversa: recupera o token correspondente a cada ID."""
        return [self.id_para_str[int(i)] for i in ids]

    def encode(self, texto: str) -> list[int]:
        """Texto -> token IDs.

        Os tokens especiais escritos literalmente no texto (como
        ``<|endoftext|>``) são preservados como um único token.
        """
        tokens: list[str] = []
        for parte in re.split(f"({'|'.join(map(re.escape, TOKENS_ESPECIAIS))})", texto):
            if parte in TOKENS_ESPECIAIS:
                tokens.append(parte)
            elif parte:
                tokens.extend(dividir_em_tokens(parte))
        return self.tokens_para_ids(tokens)

    def decode(self, ids: Iterable[int]) -> str:
        """Token IDs -> texto, reaproximando a pontuação da palavra anterior."""
        texto = " ".join(self.ids_para_tokens(ids))
        return _ESPACO_ANTES_DA_PONTUACAO.sub(r"\1", texto)


class TokenizadorBPE:
    """Tokenizador *Byte Pair Encoding* do GPT-2, sobre a biblioteca ``tiktoken``.

    O BPE parte de bytes individuais e funde, repetidamente, os pares mais
    frequentes do corpus de treinamento em unidades maiores. O resultado é um
    vocabulário de 50.257 *subpalavras*: palavras comuns viram um único token,
    palavras raras são quebradas em pedaços e, no limite, em bytes. Por isso o
    BPE **nunca** precisa de ``<|unk|>`` — qualquer texto é representável — e o
    ``decode`` reconstrói o texto original exatamente, espaços incluídos.

    O único token especial do GPT-2 é ``<|endoftext|>`` (ID 50256).
    """

    def __init__(self, codificacao: str = "gpt2") -> None:
        self.codificacao = tiktoken.get_encoding(codificacao)

    @property
    def tamanho_vocabulario(self) -> int:
        return self.codificacao.n_vocab

    def encode(self, texto: str) -> list[int]:
        """Texto -> token IDs, reconhecendo ``<|endoftext|>`` como especial."""
        return self.codificacao.encode(texto, allowed_special={FIM_DE_TEXTO})

    def decode(self, ids: Iterable[int]) -> str:
        return self.codificacao.decode([int(i) for i in ids])

    def ids_para_tokens(self, ids: Iterable[int]) -> list[str]:
        """Decodifica cada ID isoladamente — mostra como o texto foi fatiado.

        Os pedaços preservam o espaço que os precede (``" world"``), que no BPE
        do GPT-2 faz parte do próprio token.
        """
        return [self.codificacao.decode([int(i)]) for i in ids]

    def tokens_para_ids(self, tokens: Iterable[str]) -> list[int]:
        """Cada pedaço deve corresponder a exatamente um token do vocabulário."""
        return [self.codificacao.encode_single_token(t) for t in tokens]

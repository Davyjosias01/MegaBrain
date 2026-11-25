"""Fine-tuning do GPT para seguir instruções — Sprint 6 (Capítulo 7).

Na classificação (``classificacao.py``) o modelo foi adaptado para devolver
uma classe. Aqui ele continua fazendo o que sempre fez — prever o próximo
token — mas é ajustado em textos com um formato fixo, em que uma **instrução**
é seguida da **resposta** esperada. Depois do ajuste, quando recebe só a
instrução nesse formato, a continuação mais provável é uma resposta.

Nada muda na arquitetura nem na função de perda: é o mesmo treinamento da
Sprint 5, com outros dados. O trabalho deste módulo está na preparação desses
dados.

Etapas, na ordem das seções deste arquivo:

1. **dados** — carregar a base, formatar cada exemplo como um prompt e
   dividir em treino, validação e teste;
2. **lotes** — tokenizar e agrupar exemplos de tamanhos diferentes, com o
   padding fora do cálculo da perda;
3. **uso** — gerar a resposta do modelo a uma instrução;
4. **avaliação** — comparar as respostas geradas com as esperadas.

A base é a do capítulo: ``instruction-data.json``, com 1.100 exemplos em
inglês, do repositório do livro (github.com/rasbt/LLMs-from-scratch, licença
Apache 2.0).
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from geracao import gerar
from tokenizacao import TokenizadorBPE

__all__ = [
    "PASTA_INSTRUCOES",
    "MARCADOR_RESPOSTA",
    "baixar_base_instrucoes",
    "carregar_instrucoes",
    "remover_instrucoes_repetidas",
    "formatar_entrada",
    "formatar_exemplo",
    "dividir_instrucoes",
    "ID_PADDING",
    "INDICE_IGNORADO",
    "DatasetDeInstrucoes",
    "agrupar_lote",
    "preparar_instrucoes",
    "criar_loaders_instrucoes",
    "montar_prompt",
    "responder",
    "normalizar_texto",
    "correspondencia_exata",
    "f1_de_palavras",
    "avaliar_respostas",
]

RAIZ = Path(__file__).resolve().parent.parent
PASTA_INSTRUCOES = RAIZ / "sprints" / "dados" / "instrucoes"

URL_INSTRUCOES = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch07/01_main-chapter-code/instruction-data.json"
)

# Textos fixos do formato Alpaca. Ficam em inglês porque a base e o modelo
# pré-treinado são em inglês.
_ABERTURA = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
_MARCADOR_INSTRUCAO = "\n\n### Instruction:\n"
_MARCADOR_ENTRADA = "\n\n### Input:\n"
MARCADOR_RESPOSTA = "\n\n### Response:\n"

# <|endoftext|>: marca o fim da resposta e completa as sequências curtas.
ID_PADDING = 50256
# Valor que a entropia cruzada do PyTorch ignora por padrão (ignore_index).
INDICE_IGNORADO = -100


# ---------------------------------------------------------------------------
# 1. Dados: carga, formato do prompt e divisão
# ---------------------------------------------------------------------------
def baixar_base_instrucoes(pasta: Path | str = PASTA_INSTRUCOES) -> Path:
    """Baixa ``instruction-data.json`` uma única vez e devolve o caminho local."""
    destino = Path(pasta) / "instruction-data.json"
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    requisicao = urllib.request.Request(URL_INSTRUCOES, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(requisicao) as resposta:
        destino.write_bytes(resposta.read())
    return destino


def carregar_instrucoes(caminho: Path | str | None = None) -> list[dict]:
    """Lê a base: uma lista de exemplos ``{"instruction", "input", "output"}``.

    - ``instruction`` — a tarefa, em linguagem natural;
    - ``input`` — o dado sobre o qual a tarefa age; vazio em cerca de dois
      terços dos exemplos, quando a instrução já contém tudo;
    - ``output`` — a resposta esperada.
    """
    caminho = Path(caminho) if caminho is not None else baixar_base_instrucoes()
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    for exemplo in dados:
        if set(exemplo) != {"instruction", "input", "output"}:
            raise ValueError(f"exemplo com campos inesperados: {sorted(exemplo)}")
    return dados


def remover_instrucoes_repetidas(dados: list[dict]) -> list[dict]:
    """Mantém a primeira ocorrência de cada par (instrução, entrada).

    A base tem 25 pares repetidos, com respostas redigidas de forma diferente.
    Sem esta etapa, a divisão do capítulo deixa 2 perguntas do teste e 4 da
    validação também no treino — o modelo seria avaliado em perguntas cuja
    resposta ele já viu.
    """
    vistos, unicos = set(), []
    for exemplo in dados:
        chave = (exemplo["instruction"], exemplo["input"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(exemplo)
    return unicos


def formatar_entrada(exemplo: dict) -> str:
    """Monta o prompt no formato Alpaca, **sem** a resposta.

    ::

        Below is an instruction that describes a task. Write a response that
        appropriately completes the request.

        ### Instruction:
        <instrução>

        ### Input:
        <entrada>            <- bloco omitido quando a entrada é vazia

    Os marcadores ``### ...`` não são tokens especiais: são texto comum, que o
    tokenizador divide em tokens normais. O modelo aprende, pelo ajuste, que
    depois de ``### Response:`` vem a resposta — é isso que dá ao formato o
    seu papel.
    """
    texto = _ABERTURA + _MARCADOR_INSTRUCAO + exemplo["instruction"]
    if exemplo.get("input"):
        texto += _MARCADOR_ENTRADA + exemplo["input"]
    return texto


def formatar_exemplo(exemplo: dict) -> str:
    """Prompt completo de treino: entrada formatada + marcador + resposta esperada."""
    return formatar_entrada(exemplo) + MARCADOR_RESPOSTA + exemplo["output"]


def dividir_instrucoes(
    dados: list[dict], proporcao_treino: float = 0.85, proporcao_teste: float = 0.10
) -> tuple[list[dict], list[dict], list[dict]]:
    """Divide em treino, validação e teste (padrão 85% / 5% / 10%), na ordem da base.

    O teste é maior que a validação porque aqui a avaliação final é feita
    sobre respostas **geradas**, comparadas uma a uma com as esperadas; a
    validação só acompanha a perda durante o treino. O que sobra depois de
    treino e teste vai para a validação.

    Returns:
        ``(treino, validacao, teste)``.
    """
    if not 0 < proporcao_treino < 1 or not 0 <= proporcao_teste < 1 - proporcao_treino:
        raise ValueError("proporções inválidas: treino e teste devem somar menos de 1.")
    fim_treino = int(len(dados) * proporcao_treino)
    fim_teste = fim_treino + int(len(dados) * proporcao_teste)
    return dados[:fim_treino], dados[fim_teste:], dados[fim_treino:fim_teste]


# ---------------------------------------------------------------------------
# 2. Lotes: tokenização e agrupamento
# ---------------------------------------------------------------------------
class DatasetDeInstrucoes(Dataset):
    """Exemplos de instrução já formatados e tokenizados.

    Cada item é a lista de token IDs do prompt completo de treino
    (``formatar_exemplo``): instrução, entrada e resposta em uma única
    sequência. Os itens têm comprimentos diferentes; quem os iguala é
    ``agrupar_lote``, lote a lote.
    """

    def __init__(self, dados: list[dict], tokenizador=None) -> None:
        tokenizador = tokenizador or TokenizadorBPE()
        self.dados = dados
        self.textos = [tokenizador.encode(formatar_exemplo(exemplo)) for exemplo in dados]

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, idx: int) -> list[int]:
        return self.textos[idx]


def agrupar_lote(
    lote: list[list[int]],
    id_padding: int = ID_PADDING,
    indice_ignorado: int = INDICE_IGNORADO,
    max_length: int | None = None,
    dispositivo: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Função de *collate*: transforma uma lista de sequências em ``(entradas, alvos)``.

    É chamada pelo ``DataLoader`` para cada lote e faz quatro coisas:

    1. **Padding por lote.** Todas as sequências são completadas com
       ``id_padding`` até o comprimento da maior **deste lote** — não da base
       inteira, como na classificação. Lotes de exemplos curtos ficam curtos,
       e o modelo não gasta tempo processando padding.
    2. **Alvo deslocado.** Como no pré-treino, o alvo é a entrada deslocada
       uma posição para a esquerda: em cada posição, o token seguinte. Para
       isso cada sequência recebe um ``<|endoftext|>`` ao final — é ele o
       alvo do último token da resposta, e é assim que o modelo aprende a
       **parar** depois de responder.
    3. **Padding ignorado na perda.** Nos alvos, todo padding depois desse
       primeiro ``<|endoftext|>`` é trocado por ``indice_ignorado`` (-100),
       valor que ``cross_entropy`` desconsidera. Sem isso, boa parte da perda
       viria de prever padding depois de padding, o que não ensina nada.
    4. **Truncamento.** Com ``max_length``, entradas e alvos são cortados
       nesse comprimento, para não exceder o ``context_length`` do modelo.

    Exemplo, com ``id_padding = 50256``::

        lote     = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
        entradas = [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
        alvos    = [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100], [8, 9, 50256, -100, -100]]

    Returns:
        ``(entradas, alvos)``, ambos ``long`` com forma ``(lote, comprimento)``.
    """
    maior = max(len(item) + 1 for item in lote)  # +1: o <|endoftext|> final
    entradas, alvos = [], []
    for item in lote:
        completo = item + [id_padding] * (maior - len(item))
        entrada = torch.tensor(completo[:-1], dtype=torch.long)
        alvo = torch.tensor(completo[1:], dtype=torch.long)

        # Mantém o primeiro padding do alvo (o fim da resposta) e ignora os demais.
        posicoes_de_padding = torch.nonzero(alvo == id_padding).flatten()
        if posicoes_de_padding.numel() > 1:
            alvo[posicoes_de_padding[1:]] = indice_ignorado

        if max_length is not None:
            entrada, alvo = entrada[:max_length], alvo[:max_length]
        entradas.append(entrada)
        alvos.append(alvo)
    return torch.stack(entradas).to(dispositivo), torch.stack(alvos).to(dispositivo)


def preparar_instrucoes(
    pasta: Path | str = PASTA_INSTRUCOES,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Executa a seção 1 inteira: baixa, remove repetições e divide.

    Returns:
        ``(treino, validacao, teste)``, listas de exemplos.
    """
    dados = remover_instrucoes_repetidas(carregar_instrucoes(baixar_base_instrucoes(pasta)))
    return dividir_instrucoes(dados)


def criar_loaders_instrucoes(
    treino: list[dict],
    validacao: list[dict],
    teste: list[dict],
    tokenizador=None,
    batch_size: int = 8,
    max_length: int = 1024,
    semente: int = 123,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders de treino, validação e teste, com ``agrupar_lote`` como collate.

    ``max_length`` deve ser o ``context_length`` do modelo. Os lotes saem com
    comprimentos diferentes entre si (o do maior exemplo de cada um) e são
    compatíveis com ``perda_do_lote`` e ``treinar_modelo`` da Sprint 5, que
    recebem pares ``(entrada, alvo)`` e calculam a entropia cruzada — a qual
    ignora os alvos marcados com -100.

    Só o loader de treino embaralha e descarta o último lote incompleto.
    """
    tokenizador = tokenizador or TokenizadorBPE()
    collate = partial(agrupar_lote, max_length=max_length)
    gerador = torch.Generator().manual_seed(semente)
    return (
        DataLoader(DatasetDeInstrucoes(treino, tokenizador), batch_size=batch_size, shuffle=True,
                   drop_last=True, collate_fn=collate, generator=gerador),
        DataLoader(DatasetDeInstrucoes(validacao, tokenizador), batch_size=batch_size, shuffle=False,
                   drop_last=False, collate_fn=collate),
        DataLoader(DatasetDeInstrucoes(teste, tokenizador), batch_size=batch_size, shuffle=False,
                   drop_last=False, collate_fn=collate),
    )


# ---------------------------------------------------------------------------
# 3. Uso: gerar a resposta a uma instrução
# ---------------------------------------------------------------------------
def montar_prompt(instrucao: str | dict, entrada: str = "") -> str:
    """Prompt de inferência: a entrada formatada seguida de ``### Response:``.

    É o prompt de treino (``formatar_exemplo``) cortado exatamente onde a
    resposta começaria. O modelo ajustado viu centenas de textos com esse
    começo e aprendeu que o que vem a seguir é a resposta, terminada por
    ``<|endoftext|>``.

    Aceita a instrução e a entrada como textos ou um exemplo da base (o campo
    ``output``, se houver, é ignorado).
    """
    exemplo = instrucao if isinstance(instrucao, dict) else {"instruction": instrucao, "input": entrada}
    return formatar_entrada(exemplo) + MARCADOR_RESPOSTA


@torch.no_grad()
def responder(
    instrucao: str | dict,
    modelo,
    tokenizador=None,
    entrada: str = "",
    dispositivo: torch.device | str = "cpu",
    max_novos_tokens: int = 256,
    temperatura: float = 0.0,
    top_k: int | None = None,
) -> str:
    """Gera a resposta do modelo a uma instrução e devolve **só** a resposta.

    1. monta o prompt (``montar_prompt``) e o tokeniza;
    2. gera token a token com ``gerar`` (Sprint 4), parando em
       ``<|endoftext|>`` ou depois de ``max_novos_tokens``;
    3. descarta os tokens do prompt — a função de geração devolve o contexto
       seguido da continuação — e decodifica o resto.

    O padrão é a geração *greedy* (``temperatura=0``): para avaliar respostas
    interessa a saída mais provável, e ela é determinística.

    Um modelo **não** ajustado também recebe o prompt e gera uma continuação,
    mas não aprendeu a parar: sem o ``<|endoftext|>``, ele segue escrevendo
    até ``max_novos_tokens``.
    """
    tokenizador = tokenizador or TokenizadorBPE()
    ids = tokenizador.encode(montar_prompt(instrucao, entrada))
    contexto = modelo.cfg["context_length"]
    if len(ids) >= contexto:
        raise ValueError(
            f"o prompt tem {len(ids)} tokens e não cabe no contexto de {contexto} do modelo."
        )
    entrada_ids = torch.tensor(ids, dtype=torch.long, device=dispositivo).unsqueeze(0)

    treinando = modelo.training
    modelo.eval()
    saida = gerar(modelo, entrada_ids, max_novos_tokens, contexto,
                  temperatura=temperatura, top_k=top_k, id_fim=ID_PADDING)
    modelo.train(treinando)

    return tokenizador.decode(saida[0, len(ids):].tolist()).strip()


# ---------------------------------------------------------------------------
# 4. Avaliação das respostas
# ---------------------------------------------------------------------------
# Na classificação bastava comparar a classe prevista com o rótulo. Aqui a
# saída é um texto livre, e a mesma resposta correta pode ser escrita de várias
# formas ("The capital is Paris." / "Paris is the capital."). Não existe uma
# métrica automática que resolva isso por completo:
#
# - a perda de validação mede a probabilidade da resposta esperada, não a
#   qualidade do que o modelo de fato gera;
# - as métricas de sobreposição abaixo são objetivas e reproduzíveis, mas
#   medem semelhança de palavras, não correção — penalizam uma paráfrase
#   certa e premiam uma resposta errada que reaproveite as palavras da
#   pergunta;
# - a leitura humana das respostas é a referência, mas não escala;
# - o capítulo usa um segundo LLM como juiz, que dá uma nota de 0 a 100. Não
#   foi adotado aqui porque exigiria um modelo pronto de terceiros, fora da
#   proposta do projeto.
#
# O projeto combina as métricas de sobreposição com a leitura de uma amostra.
def normalizar_texto(texto: str) -> list[str]:
    """Minúsculas, sem pontuação, dividido em palavras.

    Tira da comparação as diferenças que não mudam o conteúdo: caixa, ponto
    final, aspas, espaços repetidos.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", texto.lower())


def correspondencia_exata(resposta: str, esperada: str) -> bool:
    """``True`` se as duas respostas têm as mesmas palavras, na mesma ordem (*exact match*)."""
    return normalizar_texto(resposta) == normalizar_texto(esperada)


def f1_de_palavras(resposta: str, esperada: str) -> float:
    """F1 da sobreposição de palavras entre a resposta gerada e a esperada.

    Conta as palavras em comum (com repetição, sem considerar a ordem) e
    calcula:

    - precisão: fração das palavras **geradas** que estão na esperada;
    - revocação: fração das palavras **esperadas** que foram geradas;
    - F1: média harmônica das duas.

    Vale 1 para respostas com as mesmas palavras e 0 quando não há nenhuma em
    comum. Uma resposta longa demais perde precisão; uma curta demais perde
    revocação.
    """
    geradas, esperadas = normalizar_texto(resposta), normalizar_texto(esperada)
    if not geradas or not esperadas:
        return float(geradas == esperadas)
    em_comum = sum((Counter(geradas) & Counter(esperadas)).values())
    if em_comum == 0:
        return 0.0
    precisao, revocacao = em_comum / len(geradas), em_comum / len(esperadas)
    return 2 * precisao * revocacao / (precisao + revocacao)


def avaliar_respostas(pares: list[tuple[str, str]]) -> dict[str, float]:
    """Médias das métricas sobre pares ``(resposta gerada, resposta esperada)``.

    Returns:
        ``{"exemplos", "correspondencia_exata", "f1_de_palavras", "palavras_geradas",
        "palavras_esperadas"}`` — as duas últimas são os comprimentos médios,
        que mostram se o modelo responde no tamanho certo ou não sabe parar.
    """
    if not pares:
        raise ValueError("nenhum par de respostas para avaliar.")
    n = len(pares)
    return {
        "exemplos": n,
        "correspondencia_exata": sum(correspondencia_exata(r, e) for r, e in pares) / n,
        "f1_de_palavras": sum(f1_de_palavras(r, e) for r, e in pares) / n,
        "palavras_geradas": sum(len(normalizar_texto(r)) for r, _ in pares) / n,
        "palavras_esperadas": sum(len(normalizar_texto(e)) for _, e in pares) / n,
    }

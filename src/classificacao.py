"""Fine-tuning do GPT para classificação de textos — Sprint 6 (Capítulo 6).

O GPT pré-treinado sabe prever o próximo token. Aqui ele é adaptado para outra
tarefa: receber um texto e devolver uma **classe**. A tarefa de referência é a
do capítulo — dizer se uma mensagem SMS é *spam* ou não.

Etapas, na ordem das seções deste arquivo:

1. **dados** — baixar a base, remover mensagens repetidas, balancear as
   classes e dividir em treino, validação e teste;
2. **lotes** — tokenizar e deixar todas as mensagens com o mesmo comprimento;
3. **modelo** — trocar a cabeça de saída e escolher o que fica treinável;
4. **avaliação** — perda e acurácia a partir do logit do último token;
5. **treinamento** — o laço de fine-tuning;
6. **uso** — classificar um texto novo.

A base é a *SMS Spam Collection* (Almeida e Hidalgo, UCI Machine Learning
Repository, licença CC BY 4.0): 5.574 mensagens em inglês, 747 delas *spam*.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

__all__ = [
    "PASTA_SPAM",
    "NOMES_DAS_CLASSES",
    "baixar_base_spam",
    "carregar_base_spam",
    "remover_duplicatas",
    "balancear_classes",
    "dividir_treino_validacao_teste",
    "preparar_base_spam",
]

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SPAM = RAIZ / "sprints" / "dados" / "sms-spam"

URL_SPAM = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARQUIVO_NO_ZIP = "SMSSpamCollection"

# Índice da classe -> nome. Na base original, "ham" é a mensagem legítima.
NOMES_DAS_CLASSES = ["não spam", "spam"]
_ROTULOS = {"ham": 0, "spam": 1}


# ---------------------------------------------------------------------------
# 1. Dados: download, balanceamento e divisão
# ---------------------------------------------------------------------------
def baixar_base_spam(pasta: Path | str = PASTA_SPAM) -> Path:
    """Baixa a *SMS Spam Collection* uma única vez e devolve o caminho do ``.tsv``.

    O arquivo original é um texto separado por tabulações, sem cabeçalho, com
    o rótulo (``ham`` ou ``spam``) na primeira coluna e a mensagem na segunda.
    """
    destino = Path(pasta) / "SMSSpamCollection.tsv"
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    requisicao = urllib.request.Request(URL_SPAM, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(requisicao) as resposta:
        conteudo = resposta.read()
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        destino.write_bytes(z.read(ARQUIVO_NO_ZIP))
    return destino


def carregar_base_spam(caminho: Path | str | None = None) -> pd.DataFrame:
    """Lê a base como um DataFrame com as colunas ``rotulo`` (0 ou 1) e ``texto``.

    ``quoting=3`` desliga a interpretação de aspas: as mensagens contêm aspas
    soltas, que de outro modo fariam o leitor juntar várias linhas em uma.
    """
    caminho = Path(caminho) if caminho is not None else baixar_base_spam()
    dados = pd.read_csv(
        caminho, sep="\t", header=None, names=["rotulo", "texto"], quoting=3, encoding="utf-8"
    )
    dados["rotulo"] = dados["rotulo"].map(_ROTULOS)
    if dados["rotulo"].isna().any():
        raise ValueError("a base contém rótulos diferentes de 'ham' e 'spam'.")
    dados["rotulo"] = dados["rotulo"].astype(int)
    return dados


def remover_duplicatas(dados: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma única ocorrência de cada mensagem.

    A base tem mensagens repetidas (403 das 5.574 linhas), sobretudo de spam,
    que é enviado em massa. Se uma mesma mensagem cair no treino e no teste, o
    teste deixa de medir generalização: o modelo acerta porque já viu aquele
    texto. O capítulo não faz esta etapa; aqui ela vem antes do balanceamento
    para que os três conjuntos não compartilhem nenhuma mensagem.
    """
    return dados.drop_duplicates(subset="texto").reset_index(drop=True)


def balancear_classes(dados: pd.DataFrame, semente: int = 123) -> pd.DataFrame:
    """Reduz todas as classes ao tamanho da menor, por sorteio (*undersampling*).

    Na base original, 87% das mensagens são legítimas. Um classificador que
    respondesse sempre "não spam" teria 87% de acurácia sem ter aprendido nada,
    e a perda seria dominada pela classe majoritária. Com as classes iguais, a
    acurácia de um palpite constante é 50% e qualquer valor acima disso é
    aprendizado. O custo é descartar a maior parte das mensagens legítimas.
    """
    menor = dados["rotulo"].value_counts().min()
    partes = [
        grupo.sample(menor, random_state=semente) for _, grupo in dados.groupby("rotulo")
    ]
    return pd.concat(partes).sort_index().reset_index(drop=True)


def dividir_treino_validacao_teste(
    dados: pd.DataFrame,
    proporcao_treino: float = 0.7,
    proporcao_validacao: float = 0.1,
    semente: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Embaralha e divide em treino, validação e teste (padrão 70% / 10% / 20%).

    - **treino**: os exemplos que ajustam os pesos;
    - **validação**: acompanhada durante o treino, para escolher configurações
      e perceber sobreajuste;
    - **teste**: usado uma única vez, no fim, para medir o desempenho em dados
      que não influenciaram nenhuma decisão.

    O que sobra depois de treino e validação vai para o teste.
    """
    if not 0 < proporcao_treino < 1 or not 0 <= proporcao_validacao < 1 - proporcao_treino:
        raise ValueError("proporções inválidas: treino e validação devem somar menos de 1.")
    embaralhado = dados.sample(frac=1, random_state=semente).reset_index(drop=True)
    fim_treino = int(len(embaralhado) * proporcao_treino)
    fim_validacao = fim_treino + int(len(embaralhado) * proporcao_validacao)
    return (
        embaralhado[:fim_treino].reset_index(drop=True),
        embaralhado[fim_treino:fim_validacao].reset_index(drop=True),
        embaralhado[fim_validacao:].reset_index(drop=True),
    )


def preparar_base_spam(
    pasta: Path | str = PASTA_SPAM, semente: int = 123
) -> dict[str, Path]:
    """Executa a seção 1 inteira e grava ``treino.csv``, ``validacao.csv`` e ``teste.csv``.

    Returns:
        Dicionário ``{"treino": ..., "validacao": ..., "teste": ...}`` com os caminhos.
    """
    pasta = Path(pasta)
    dados = remover_duplicatas(carregar_base_spam(baixar_base_spam(pasta)))
    dados = balancear_classes(dados, semente)
    partes = dividir_treino_validacao_teste(dados, semente=semente)
    caminhos = {}
    for nome, parte in zip(("treino", "validacao", "teste"), partes):
        caminhos[nome] = pasta / f"{nome}.csv"
        parte.to_csv(caminhos[nome], index=False, encoding="utf-8", lineterminator="\n")
    return caminhos

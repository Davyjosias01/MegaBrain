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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gpt import GPTModel
from tokenizacao import TokenizadorBPE

__all__ = [
    "PASTA_SPAM",
    "NOMES_DAS_CLASSES",
    "baixar_base_spam",
    "carregar_base_spam",
    "remover_duplicatas",
    "balancear_classes",
    "dividir_treino_validacao_teste",
    "preparar_base_spam",
    "ID_PADDING",
    "DatasetDeClassificacao",
    "criar_loaders_classificacao",
    "adaptar_para_classificacao",
    "contar_treinaveis",
    "logits_de_classe",
    "perda_classificacao_do_lote",
    "perda_classificacao_do_loader",
    "acuracia_do_loader",
    "treinar_classificador",
    "classificar_texto",
]

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SPAM = RAIZ / "sprints" / "dados" / "sms-spam"

URL_SPAM = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARQUIVO_NO_ZIP = "SMSSpamCollection"

# Índice da classe -> nome. Na base original, "ham" é a mensagem legítima.
NOMES_DAS_CLASSES = ["não spam", "spam"]
_ROTULOS = {"ham": 0, "spam": 1}

# <|endoftext|>, o token usado para completar as mensagens curtas.
ID_PADDING = 50256


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


# ---------------------------------------------------------------------------
# 2. Lotes: tokenização, truncamento e padding
# ---------------------------------------------------------------------------
class DatasetDeClassificacao(Dataset):
    """Mensagens tokenizadas, todas com o mesmo comprimento, e seus rótulos.

    Um lote é um tensor retangular, mas as mensagens têm tamanhos diferentes.
    A solução do capítulo é levar todas a um comprimento único:

    - mensagens **maiores** que ``max_length`` são truncadas;
    - mensagens **menores** são completadas à direita com ``id_padding``
      (``<|endoftext|>``, 50256).

    Se ``max_length`` não for informado, vale o comprimento da maior mensagem
    deste conjunto. Os conjuntos de validação e de teste devem receber o
    ``max_length`` do **treino**: o modelo é ajustado para sequências desse
    tamanho, e o comprimento não pode depender de dados que ele não viu.

    Cada item é ``(ids, rotulo)``: ``ids`` com forma ``(max_length,)`` e
    ``rotulo`` um inteiro (tensor escalar ``long``).
    """

    def __init__(
        self,
        dados: pd.DataFrame | Path | str,
        tokenizador=None,
        max_length: int | None = None,
        id_padding: int = ID_PADDING,
    ) -> None:
        if not isinstance(dados, pd.DataFrame):
            dados = pd.read_csv(dados, encoding="utf-8")
        tokenizador = tokenizador or TokenizadorBPE()
        self.rotulos = dados["rotulo"].astype(int).tolist()
        codificados = [tokenizador.encode(texto) for texto in dados["texto"]]
        self.max_length = max_length or max(len(ids) for ids in codificados)
        self.id_padding = id_padding
        self.textos = [
            ids[: self.max_length] + [id_padding] * (self.max_length - len(ids))
            for ids in codificados
        ]

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.textos[idx], dtype=torch.long),
            torch.tensor(self.rotulos[idx], dtype=torch.long),
        )


def criar_loaders_classificacao(
    pasta: Path | str = PASTA_SPAM,
    tokenizador=None,
    batch_size: int = 8,
    max_length: int | None = None,
    semente: int = 123,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders de treino, validação e teste a partir dos CSV de ``preparar_base_spam``.

    O comprimento das sequências é o do conjunto de **treino** (ou
    ``max_length``, se informado — por exemplo o ``context_length`` do modelo,
    quando há mensagens maiores que ele) e é aplicado aos três conjuntos.

    Só o loader de treino embaralha e descarta o último lote incompleto; os
    de validação e teste percorrem todos os exemplos, sempre na mesma ordem.
    """
    pasta = Path(pasta)
    tokenizador = tokenizador or TokenizadorBPE()
    treino = DatasetDeClassificacao(pasta / "treino.csv", tokenizador, max_length)
    validacao = DatasetDeClassificacao(pasta / "validacao.csv", tokenizador, treino.max_length)
    teste = DatasetDeClassificacao(pasta / "teste.csv", tokenizador, treino.max_length)
    gerador = torch.Generator().manual_seed(semente)
    return (
        DataLoader(treino, batch_size=batch_size, shuffle=True, drop_last=True, generator=gerador),
        DataLoader(validacao, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(teste, batch_size=batch_size, shuffle=False, drop_last=False),
    )


# ---------------------------------------------------------------------------
# 3. Modelo: cabeça de classificação e congelamento
# ---------------------------------------------------------------------------
def adaptar_para_classificacao(
    modelo: GPTModel,
    num_classes: int = 2,
    blocos_treinaveis: int | None = 1,
    treinar_norm_final: bool = True,
) -> GPTModel:
    """Transforma um GPT gerador de texto em um classificador.

    Duas alterações, feitas no próprio ``modelo``:

    1. **Cabeça de classificação.** ``out_head`` deixa de ser
       ``Linear(emb_dim, vocab_size)`` — uma pontuação para cada um dos 50.257
       tokens — e passa a ser ``Linear(emb_dim, num_classes)`` — uma pontuação
       por classe. É a única parte nova do modelo: começa com pesos aleatórios
       e precisa ser treinada. Todo o resto continua com os pesos que já tinha.
    2. **Congelamento.** ``requires_grad = False`` em um parâmetro faz o
       PyTorch não calcular gradiente para ele: o otimizador não o altera.
       Aqui todos os parâmetros são congelados e só voltam a ser treináveis a
       cabeça nova, os últimos ``blocos_treinaveis`` blocos Transformer e,
       se ``treinar_norm_final``, a ``final_norm``.

    Por que congelar: as primeiras camadas de um modelo pré-treinado guardam
    conhecimento geral da língua, que serve a qualquer tarefa; as últimas são
    as mais específicas da tarefa original. Ajustar só o fim preserva o que o
    pré-treino aprendeu, custa menos (menos gradientes a calcular e a guardar)
    e reduz o sobreajuste em uma base de mil exemplos.

    Args:
        blocos_treinaveis: quantos blocos, contados a partir do último, ficam
            treináveis. ``0`` treina só a cabeça (e a norma final); ``None``
            descongela o modelo inteiro (*full fine-tuning*).

    Returns:
        O próprio ``modelo``, com ``modelo.cfg["num_classes"]`` registrado.
    """
    n_blocos = len(modelo.trf_blocks)
    if blocos_treinaveis is not None and not 0 <= blocos_treinaveis <= n_blocos:
        raise ValueError(f"blocos_treinaveis deve estar entre 0 e {n_blocos}.")

    tudo_treinavel = blocos_treinaveis is None
    for parametro in modelo.parameters():
        parametro.requires_grad = tudo_treinavel

    dispositivo = next(modelo.parameters()).device
    modelo.out_head = nn.Linear(modelo.cfg["emb_dim"], num_classes).to(dispositivo)
    modelo.cfg["num_classes"] = num_classes

    if not tudo_treinavel:
        for bloco in list(modelo.trf_blocks)[n_blocos - blocos_treinaveis:]:
            for parametro in bloco.parameters():
                parametro.requires_grad = True
        for parametro in modelo.final_norm.parameters():
            parametro.requires_grad = treinar_norm_final
    return modelo


def contar_treinaveis(modelo: nn.Module) -> tuple[int, int]:
    """``(parâmetros treináveis, parâmetros totais)`` do modelo."""
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return treinaveis, total


# ---------------------------------------------------------------------------
# 4. Avaliação: perda e acurácia
# ---------------------------------------------------------------------------
def logits_de_classe(
    modelo: nn.Module, entrada: torch.Tensor, posicao: int = -1
) -> torch.Tensor:
    """``(b, L)`` token IDs -> ``(b, num_classes)``: os logits de **uma** posição.

    O modelo adaptado produz ``num_classes`` logits para cada token da
    sequência, ``(b, L, num_classes)``, mas a mensagem tem um único rótulo.
    Usa-se a saída do **último** token (``posicao=-1``): por causa da máscara
    causal da Sprint 3, cada posição só enxerga a si mesma e as anteriores, de
    modo que a última é a única cujo vetor foi calculado a partir da mensagem
    inteira. A primeira posição, ao contrário, só enxerga o primeiro token.

    Nas mensagens curtas a última posição é um token de padding. Isso não
    atrapalha: o padding fica à direita, então essa posição continua
    enxergando todos os tokens reais que vieram antes.
    """
    return modelo(entrada)[:, posicao, :]


def perda_classificacao_do_lote(
    entrada: torch.Tensor,
    alvo: torch.Tensor,
    modelo: nn.Module,
    dispositivo: torch.device | str,
    posicao: int = -1,
) -> torch.Tensor:
    """Entropia cruzada entre os logits de classe e os rótulos do lote.

    É a mesma função de perda do treinamento da Sprint 5. O que muda é o que
    está sendo previsto: lá, um token entre 50.257 em cada posição; aqui, uma
    classe entre ``num_classes`` para a mensagem inteira.
    """
    entrada, alvo = entrada.to(dispositivo), alvo.to(dispositivo)
    return nn.functional.cross_entropy(logits_de_classe(modelo, entrada, posicao), alvo)


@torch.no_grad()
def perda_classificacao_do_loader(
    loader,
    modelo: nn.Module,
    dispositivo: torch.device | str,
    num_lotes: int | None = None,
    posicao: int = -1,
) -> float:
    """Perda média sobre os lotes do loader (os ``num_lotes`` primeiros, se informado).

    Roda em ``eval()`` e devolve o modelo ao modo em que estava. Loader vazio
    resulta em ``nan``.
    """
    if len(loader) == 0:
        return float("nan")
    num_lotes = len(loader) if num_lotes is None else min(num_lotes, len(loader))
    treinando = modelo.training
    modelo.eval()
    total = 0.0
    for i, (entrada, alvo) in enumerate(loader):
        if i >= num_lotes:
            break
        total += perda_classificacao_do_lote(entrada, alvo, modelo, dispositivo, posicao).item()
    modelo.train(treinando)
    return total / num_lotes


@torch.no_grad()
def acuracia_do_loader(
    loader,
    modelo: nn.Module,
    dispositivo: torch.device | str,
    num_lotes: int | None = None,
    posicao: int = -1,
) -> float:
    """Fração de exemplos classificados corretamente (*classification accuracy*).

    A classe prevista é a de maior logit (``argmax``). Não é preciso aplicar o
    softmax: ele preserva a ordem dos logits. A acurácia é a métrica que
    interessa, mas não serve de função de perda porque não é diferenciável —
    uma mudança pequena nos pesos quase nunca muda o ``argmax``, então o
    gradiente seria zero. Por isso treina-se com a entropia cruzada e mede-se
    com a acurácia.
    """
    if len(loader) == 0:
        return float("nan")
    num_lotes = len(loader) if num_lotes is None else min(num_lotes, len(loader))
    treinando = modelo.training
    modelo.eval()
    corretos, exemplos = 0, 0
    for i, (entrada, alvo) in enumerate(loader):
        if i >= num_lotes:
            break
        entrada, alvo = entrada.to(dispositivo), alvo.to(dispositivo)
        previsto = logits_de_classe(modelo, entrada, posicao).argmax(dim=-1)
        corretos += (previsto == alvo).sum().item()
        exemplos += alvo.numel()
    modelo.train(treinando)
    return corretos / exemplos


# ---------------------------------------------------------------------------
# 5. Treinamento: o laço de fine-tuning
# ---------------------------------------------------------------------------
def treinar_classificador(
    modelo: nn.Module,
    loader_treino,
    loader_validacao,
    otimizador: torch.optim.Optimizer,
    dispositivo: torch.device | str,
    num_epocas: int,
    frequencia_avaliacao: int = 50,
    lotes_avaliacao: int | None = 5,
    posicao: int = -1,
    verboso: bool = True,
) -> dict[str, list]:
    """Laço de fine-tuning supervisionado — ``train_classifier_simple`` da seção 6.7.

    A estrutura é a de ``treinar_modelo`` da Sprint 5 — para cada época e cada
    lote: ``zero_grad`` → perda → ``backward`` → ``step``. As diferenças são
    a perda (de classificação, sobre o último token), o que se conta (exemplos
    em vez de tokens) e o que se acompanha (além da perda, a acurácia).

    Só os parâmetros com ``requires_grad=True`` recebem gradiente e são
    atualizados; o que foi congelado em ``adaptar_para_classificacao`` passa
    pelo laço inalterado.

    A cada ``frequencia_avaliacao`` passos registra as perdas de treino e de
    validação em ``lotes_avaliacao`` lotes; ao fim de cada época, as acurácias
    de treino e de validação nos mesmos ``lotes_avaliacao`` lotes (``None``
    usa o conjunto inteiro).

    Returns:
        Histórico com as listas ``passos``, ``exemplos_vistos``,
        ``perda_treino``, ``perda_validacao`` (uma entrada por avaliação) e
        ``acuracia_treino``, ``acuracia_validacao`` (uma entrada por época).
    """
    historico: dict[str, list] = {
        "passos": [], "exemplos_vistos": [], "perda_treino": [], "perda_validacao": [],
        "acuracia_treino": [], "acuracia_validacao": [],
    }
    exemplos_vistos, passo = 0, -1

    for epoca in range(num_epocas):
        modelo.train()
        for entrada, alvo in loader_treino:
            otimizador.zero_grad()
            perda = perda_classificacao_do_lote(entrada, alvo, modelo, dispositivo, posicao)
            perda.backward()
            otimizador.step()
            exemplos_vistos += entrada.shape[0]
            passo += 1

            if passo % frequencia_avaliacao == 0:
                perda_treino = perda_classificacao_do_loader(
                    loader_treino, modelo, dispositivo, lotes_avaliacao, posicao)
                perda_validacao = perda_classificacao_do_loader(
                    loader_validacao, modelo, dispositivo, lotes_avaliacao, posicao)
                historico["passos"].append(passo)
                historico["exemplos_vistos"].append(exemplos_vistos)
                historico["perda_treino"].append(perda_treino)
                historico["perda_validacao"].append(perda_validacao)
                if verboso:
                    print(f"Época {epoca + 1} (passo {passo:05d}): "
                          f"perda de treino {perda_treino:.3f}, "
                          f"perda de validação {perda_validacao:.3f}")

        acuracia_treino = acuracia_do_loader(
            loader_treino, modelo, dispositivo, lotes_avaliacao, posicao)
        acuracia_validacao = acuracia_do_loader(
            loader_validacao, modelo, dispositivo, lotes_avaliacao, posicao)
        historico["acuracia_treino"].append(acuracia_treino)
        historico["acuracia_validacao"].append(acuracia_validacao)
        if verboso:
            print(f"Época {epoca + 1}: acurácia de treino {acuracia_treino:.2%}, "
                  f"acurácia de validação {acuracia_validacao:.2%}")

    modelo.eval()
    return historico


# ---------------------------------------------------------------------------
# 6. Uso: classificar um texto novo
# ---------------------------------------------------------------------------
@torch.no_grad()
def classificar_texto(
    texto: str,
    modelo: nn.Module,
    tokenizador=None,
    dispositivo: torch.device | str = "cpu",
    max_length: int | None = None,
    id_padding: int = ID_PADDING,
    nomes: list[str] | None = None,
    posicao: int = -1,
) -> tuple[str, dict[str, float]]:
    """Classifica um texto e devolve ``(classe, probabilidades)``.

    O texto passa pelo mesmo preparo dos dados de treino: tokenização,
    truncamento e padding até ``max_length``. Informe o ``max_length`` usado
    no treino (``loader.dataset.max_length``): o modelo foi ajustado lendo a
    classe em uma posição precedida de padding, e convém repetir essa condição
    na inferência. O comprimento nunca passa do ``context_length`` do modelo.

    O softmax transforma os dois logits em probabilidades que somam 1. Elas
    indicam a confiança do modelo na previsão, mas não são calibradas: 0,99
    não significa que ele acerta 99% das vezes em que diz isso.

    >>> classe, probabilidades = classificar_texto("You won a prize!", modelo, max_length=188)
    >>> classe, probabilidades                               # doctest: +SKIP
    ('spam', {'não spam': 0.03, 'spam': 0.97})
    """
    tokenizador = tokenizador or TokenizadorBPE()
    nomes = nomes or NOMES_DAS_CLASSES
    contexto = modelo.cfg["context_length"]
    limite = min(max_length, contexto) if max_length is not None else contexto

    ids = tokenizador.encode(texto)[:limite]
    if not ids:
        raise ValueError("o texto está vazio.")
    if max_length is not None:  # sem max_length, o texto vai sem padding
        ids = ids + [id_padding] * (limite - len(ids))
    entrada = torch.tensor(ids, dtype=torch.long, device=dispositivo).unsqueeze(0)

    treinando = modelo.training
    modelo.eval()
    probabilidades = torch.softmax(logits_de_classe(modelo, entrada, posicao)[0], dim=-1)
    modelo.train(treinando)

    if len(nomes) != probabilidades.numel():
        raise ValueError(
            f"o modelo tem {probabilidades.numel()} classes e foram dados {len(nomes)} nomes."
        )
    classe = nomes[int(probabilidades.argmax())]
    return classe, {nome: float(p) for nome, p in zip(nomes, probabilidades)}

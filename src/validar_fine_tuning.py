"""Validação dos pesos pré-treinados e do fine-tuning — Sprint 6.

Confere ``pesos_pretreinados.py``, ``classificacao.py`` e ``instrucoes.py``
contra valores calculados à mão e contra as propriedades que cada etapa deve
satisfazer. Reúne, num só lugar, os testes usados em cada task da sprint
(blocos A a E do plano).

Os modelos são pequenos, com pesos aleatórios, e rodam em CPU em menos de um
minuto: o que se confere aqui é o **código**, não a qualidade dos modelos
ajustados, que está nos experimentos. Com ``--completo`` também é carregado o
GPT-2 124M real (baixa o arquivo de 548 MB se ele não estiver em ``modelos/``).

Execução, a partir da raiz do repositório::

    uv run python src/validar_fine_tuning.py
    uv run python src/validar_fine_tuning.py --completo
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classificacao import (  # noqa: E402
    ID_PADDING,
    NOMES_DAS_CLASSES,
    PASTA_SPAM,
    DatasetDeClassificacao,
    acuracia_do_loader,
    adaptar_para_classificacao,
    balancear_classes,
    carregar_base_spam,
    classificar_texto,
    contar_treinaveis,
    logits_de_classe,
    metricas_da_matriz,
    metricas_de_classificacao,
    perda_classificacao_do_loader,
    perda_classificacao_do_lote,
    prever_loader,
    remover_duplicatas,
    treinar_classificador,
)
from gpt import GPTModel, config_gpt2  # noqa: E402
from instrucoes import (  # noqa: E402
    INDICE_IGNORADO,
    MARCADOR_RESPOSTA,
    DatasetDeInstrucoes,
    agrupar_lote,
    avaliar_respostas,
    carregar_instrucoes,
    correspondencia_exata,
    criar_loaders_instrucoes,
    definir_treinaveis,
    dividir_instrucoes,
    f1_de_palavras,
    formatar_entrada,
    formatar_exemplo,
    montar_prompt,
    normalizar_texto,
    remover_instrucoes_repetidas,
    responder,
)
from pesos_pretreinados import (  # noqa: E402
    carregar_gpt2_pretreinado,
    carregar_pesos_gpt2,
    config_gpt2_pretreinado,
    mapa_de_parametros,
)
from tokenizacao import TokenizadorBPE  # noqa: E402
from treinamento import perda_do_lote, treinar_modelo  # noqa: E402

V = 50257
TK = TokenizadorBPE()
_resultados: list[tuple[bool, str]] = []


def _verificar(condicao: bool, descricao: str) -> None:
    _resultados.append((bool(condicao), descricao))
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}")


def _levanta(funcao, erro=ValueError) -> bool:
    try:
        funcao()
    except erro:
        return True
    return False


def _cfg_pequena(**alteracoes) -> dict:
    cfg = config_gpt2("small", emb_dim=64, n_heads=4, n_layers=2, context_length=64,
                      drop_rate=0.0, qkv_bias=True)
    cfg.update(alteracoes)
    return cfg


def _modelo_pequeno(semente: int = 123, **alteracoes) -> GPTModel:
    torch.manual_seed(semente)
    return GPTModel(_cfg_pequena(**alteracoes))


# ---------------------------------------------------------------------------
# A. Pesos pré-treinados
# ---------------------------------------------------------------------------
def _arquivo_sintetico(cfg: dict) -> dict[str, torch.Tensor]:
    """Tensores aleatórios com os nomes e as formas do arquivo do GPT-2."""
    torch.manual_seed(7)
    d, n = cfg["emb_dim"], cfg["context_length"]
    r = lambda *forma: torch.randn(*forma) * 0.1  # noqa: E731
    p = {"wte.weight": r(V, d), "wpe.weight": r(n, d), "ln_f.weight": 1 + r(d), "ln_f.bias": r(d)}
    for i in range(cfg["n_layers"]):
        h = f"h.{i}"
        p.update({
            f"{h}.ln_1.weight": 1 + r(d), f"{h}.ln_1.bias": r(d),
            f"{h}.attn.c_attn.weight": r(d, 3 * d), f"{h}.attn.c_attn.bias": r(3 * d),
            f"{h}.attn.c_proj.weight": r(d, d), f"{h}.attn.c_proj.bias": r(d),
            f"{h}.ln_2.weight": 1 + r(d), f"{h}.ln_2.bias": r(d),
            f"{h}.mlp.c_fc.weight": r(d, 4 * d), f"{h}.mlp.c_fc.bias": r(4 * d),
            f"{h}.mlp.c_proj.weight": r(4 * d, d), f"{h}.mlp.c_proj.bias": r(d),
            f"{h}.attn.bias": torch.tril(torch.ones(n, n)),  # máscara: deve ser ignorada
        })
    return p


def _gpt2_de_referencia(p: dict, ids: torch.Tensor, n_layers: int, n_heads: int) -> torch.Tensor:
    """Passagem direta do GPT-2 escrita sobre os tensores do arquivo.

    Usa a convenção do arquivo (``x @ W + b``, Q/K/V num tensor só) e não usa
    nenhuma classe de ``gpt.py``: se a tabela de correspondência trocar,
    transpuser ou fatiar errado algum tensor, os logits deixam de coincidir.
    """
    def norma(x, nome):
        x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
        return x * p[f"{nome}.weight"] + p[f"{nome}.bias"]

    lote, t = ids.shape
    x = p["wte.weight"][ids] + p["wpe.weight"][:t]
    d = x.shape[-1]
    mascara = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
    for i in range(n_layers):
        h = f"h.{i}"
        a = norma(x, f"{h}.ln_1") @ p[f"{h}.attn.c_attn.weight"] + p[f"{h}.attn.c_attn.bias"]
        q, k, v = (parte.view(lote, t, n_heads, d // n_heads).transpose(1, 2) for parte in a.split(d, -1))
        pesos = (q @ k.transpose(-1, -2) / math.sqrt(d // n_heads)).masked_fill(mascara, -torch.inf)
        ctx = (torch.softmax(pesos, -1) @ v).transpose(1, 2).reshape(lote, t, d)
        x = x + ctx @ p[f"{h}.attn.c_proj.weight"] + p[f"{h}.attn.c_proj.bias"]
        m = norma(x, f"{h}.ln_2") @ p[f"{h}.mlp.c_fc.weight"] + p[f"{h}.mlp.c_fc.bias"]
        m = 0.5 * m * (1 + torch.tanh(math.sqrt(2 / math.pi) * (m + 0.044715 * m**3)))
        x = x + m @ p[f"{h}.mlp.c_proj.weight"] + p[f"{h}.mlp.c_proj.bias"]
    return norma(x, "ln_f") @ p["wte.weight"].T


def teste_pesos() -> None:
    """Capítulo 5, seção 5.5 — carregamento dos pesos do GPT-2 (bloco A)."""
    print("\nPesos pré-treinados (bloco A)")
    cfg = _cfg_pequena()
    modelo = GPTModel(cfg).eval()
    mapa = mapa_de_parametros(cfg)
    destinos = [c.destino for c in mapa]
    parametros = [n for n, _ in modelo.named_parameters()]
    _verificar(sorted(destinos) == sorted(parametros) and len(set(destinos)) == len(destinos),
               f"a tabela cobre cada um dos {len(parametros)} parâmetros do modelo exatamente uma vez")

    arquivo = _arquivo_sintetico(cfg)
    carregar_pesos_gpt2(modelo, arquivo)
    sd = modelo.state_dict()
    w = arquivo["h.0.attn.c_attn.weight"]
    _verificar(all(torch.equal(sd[f"trf_blocks.0.att.{nome}.weight"], w[:, i * 64:(i + 1) * 64].T)
                   for i, nome in enumerate(("W_query", "W_key", "W_value"))),
               "c_attn é dividido em Q, K e V (terços das colunas) e transposto")
    _verificar(torch.equal(sd["out_head.weight"], arquivo["wte.weight"]),
               "a cabeça de saída recebe a matriz de embeddings (weight tying)")
    ids = torch.randint(0, V, (2, 10))
    with torch.no_grad():
        dif = (modelo(ids) - _gpt2_de_referencia(arquivo, ids, cfg["n_layers"], cfg["n_heads"])).abs().max().item()
    _verificar(dif < 1e-4, f"logits iguais aos de um GPT-2 escrito direto sobre o arquivo (diferença {dif:.1e})")

    antes = {n: t.clone() for n, t in GPTModel(_cfg_pequena(qkv_bias=False)).state_dict().items()}
    sem_bias = GPTModel(_cfg_pequena(qkv_bias=False))
    sem_bias.load_state_dict(antes)
    _verificar(_levanta(lambda: carregar_pesos_gpt2(sem_bias, arquivo)),
               "qkv_bias=False -> ValueError (os bias de c_attn ficariam sem destino)")
    curto = GPTModel(_cfg_pequena(context_length=32))
    antes = {n: t.clone() for n, t in curto.state_dict().items()}
    erro = _levanta(lambda: carregar_pesos_gpt2(curto, arquivo))
    intacto = all(torch.equal(t, antes[n]) for n, t in curto.state_dict().items())
    _verificar(erro and intacto, "forma incompatível -> ValueError, e nenhum peso do modelo é alterado")
    cfg_real = config_gpt2_pretreinado("small")
    _verificar(cfg_real["context_length"] == 1024 and cfg_real["qkv_bias"] and cfg_real["emb_dim"] == 768,
               "configuração do GPT-2 124M: contexto 1024, qkv_bias=True, emb_dim 768")


def teste_pesos_reais() -> None:
    """GPT-2 124M publicado: só com ``--completo``."""
    print("\nGPT-2 124M real (opção --completo)")
    modelo = carregar_gpt2_pretreinado("small")
    ids = torch.tensor([TK.encode("Every effort moves you")])
    with torch.no_grad():
        probs = torch.softmax(modelo(ids)[0, -1], -1)
    proximo = TK.decode([int(probs.argmax())])
    _verificar(proximo == " forward", f"token mais provável após 'Every effort moves you': {proximo!r}")
    _verificar(sum(p.numel() for p in modelo.parameters()) - modelo.out_head.weight.numel() == 124_439_808,
               "124.439.808 parâmetros, descontada a cabeça de saída (compartilhada no GPT-2)")


# ---------------------------------------------------------------------------
# B e C. Fine-tuning de classificação
# ---------------------------------------------------------------------------
def teste_dados_de_classificacao() -> None:
    """Seções 6.2 e 6.3 — base de spam e dataset (B1–B3)."""
    print("\nBase de spam e dataset (seções 6.2 e 6.3)")
    bruto = carregar_base_spam()
    unico = remover_duplicatas(bruto)
    balanceado = balancear_classes(unico)
    contagem = balanceado["rotulo"].value_counts()
    _verificar((len(bruto), len(unico), len(balanceado)) == (5574, 5171, 1306),
               f"mensagens: {len(bruto)} lidas -> {len(unico)} únicas -> {len(balanceado)} após balancear")
    _verificar(contagem[0] == contagem[1] == 653, "classes balanceadas: 653 de cada")
    partes = [pd.read_csv(PASTA_SPAM / f"{nome}.csv") for nome in ("treino", "validacao", "teste")]
    _verificar([len(p) for p in partes] == [914, 130, 262], "divisão 70/10/20: 914 / 130 / 262")
    textos = [set(p["texto"]) for p in partes]
    _verificar(not (textos[0] & textos[1] or textos[0] & textos[2] or textos[1] & textos[2]),
               "nenhuma mensagem aparece em mais de um conjunto")

    ds = DatasetDeClassificacao(partes[0].head(50), TK)
    maior = max(len(TK.encode(t)) for t in partes[0].head(50)["texto"])
    ids, rotulo = ds[0]
    n = len(TK.encode(partes[0]["texto"][0]))
    _verificar(ds.max_length == maior and all(len(ds[i][0]) == maior for i in range(len(ds))),
               f"todas as mensagens ficam com o comprimento da maior ({maior} tokens)")
    _verificar(ids[:n].tolist() == TK.encode(partes[0]["texto"][0]) and (ids[n:] == ID_PADDING).all()
               and int(rotulo) == int(partes[0]["rotulo"][0]),
               "item = tokens da mensagem + padding com <|endoftext|>, com o rótulo do arquivo")
    cortado = DatasetDeClassificacao(partes[0].head(50), TK, max_length=10)
    _verificar(all(len(cortado[i][0]) == 10 for i in range(len(cortado))), "max_length trunca as mensagens longas")


def teste_modelo_de_classificacao() -> None:
    """Seções 6.5 e 6.6 — cabeça, congelamento, perda e acurácia (B4–B5)."""
    print("\nCabeça de classificação, perda e acurácia (seções 6.5 e 6.6)")
    m = _modelo_pequeno()
    bloco = sum(p.numel() for p in m.trf_blocks[-1].parameters())
    adaptar_para_classificacao(m, 2, blocos_treinaveis=1)
    treinaveis, total = contar_treinaveis(m)
    _verificar(tuple(m.out_head.weight.shape) == (2, 64), "a cabeça passa a ser Linear(emb_dim, 2)")
    _verificar(treinaveis == bloco + 2 * 64 + (64 * 2 + 2),
               f"treináveis = último bloco + norma final + cabeça ({treinaveis:,} de {total:,})")
    _verificar(contar_treinaveis(adaptar_para_classificacao(_modelo_pequeno(), 2, 0, False))[0] == 130,
               "blocos_treinaveis=0 sem a norma final: só os 130 parâmetros da cabeça")

    m.eval()
    x = torch.randint(0, V, (6, 12))
    y = torch.tensor([0, 1, 1, 0, 1, 0])
    with torch.no_grad():
        ultimo = m(x)[:, -1, :]
        manual = -torch.log_softmax(ultimo, -1)[torch.arange(6), y].mean()
        perda = perda_classificacao_do_lote(x, y, m, "cpu")
        prev = ultimo.argmax(-1)
    _verificar(torch.allclose(perda, manual), f"perda do lote = entropia cruzada no último token ({perda:.4f})")
    _verificar(torch.allclose(logits_de_classe(m, x, 0), m(x)[:, 0, :]), "posicao=0 lê a saída do primeiro token")
    loader = lambda alvo: DataLoader(TensorDataset(x, alvo), batch_size=4)  # noqa: E731
    misto = torch.cat([prev[:3], 1 - prev[3:]])
    _verificar((acuracia_do_loader(loader(prev), m, "cpu"), acuracia_do_loader(loader(1 - prev), m, "cpu"),
                acuracia_do_loader(loader(misto), m, "cpu")) == (1.0, 0.0, 0.5),
               "acurácia: 1,0 com rótulos certos, 0,0 com trocados, 0,5 com metade")
    medias = [perda_classificacao_do_lote(a, b, m, "cpu").item() for a, b in loader(y)]
    _verificar(abs(perda_classificacao_do_loader(loader(y), m, "cpu") - sum(medias) / len(medias)) < 1e-6,
               "perda do loader = média das perdas dos lotes")
    m.train()
    acuracia_do_loader(loader(y), m, "cpu")
    _verificar(m.training, "a avaliação devolve o modelo ao modo de treino")


def teste_metricas_de_classificacao() -> None:
    """Avaliação do classificador — matriz de confusão e métricas (C2)."""
    print("\nMétricas do classificador")
    r = metricas_da_matriz(3, 1, 4, 2)
    _verificar(abs(r["acuracia"] - 0.7) < 1e-12 and abs(r["precisao"] - 0.75) < 1e-12
               and abs(r["revocacao"] - 0.6) < 1e-12 and abs(r["f1"] - 2 * 0.75 * 0.6 / 1.35) < 1e-12,
               "vp=3 fp=1 vn=4 fn=2: acurácia 0,70, precisão 0,75, revocação 0,60, F1 0,667")
    zero = metricas_da_matriz(0, 0, 5, 5)
    _verificar(zero["precisao"] == 0.0 and zero["f1"] == 0.0, "sem previsões positivas: precisão e F1 = 0, sem erro")

    m = adaptar_para_classificacao(_modelo_pequeno()).eval()
    x = torch.randint(0, V, (20, 12))
    with torch.no_grad():
        prev = m(x)[:, -1, :].argmax(-1)
    y = prev.clone()
    y[:5] = 1 - y[:5]
    loader = DataLoader(TensorDataset(x, y), batch_size=6)
    r = metricas_de_classificacao(loader, m, "cpu")
    previstos, alvos = prever_loader(loader, m, "cpu")
    vp = sum(a == 1 and b == 1 for a, b in zip(prev.tolist(), y.tolist()))
    fp = sum(a == 1 and b == 0 for a, b in zip(prev.tolist(), y.tolist()))
    _verificar(previstos == prev.tolist() and alvos == y.tolist(), "previsões e rótulos devolvidos na ordem do loader")
    _verificar(r["vp"] == vp and r["fp"] == fp and r["vp"] + r["fp"] + r["vn"] + r["fn"] == 20
               and abs(r["acuracia"] - 0.75) < 1e-12,
               "matriz de confusão confere com a contagem manual (5 erros em 20 = 75%)")


def teste_treino_de_classificacao() -> None:
    """Seção 6.7 e 6.8 — laço de treino e uso do classificador (B6, C1)."""
    print("\nTreino e uso do classificador (seções 6.7 e 6.8)")
    treino = pd.read_csv(PASTA_SPAM / "treino.csv").head(160)
    ds = DatasetDeClassificacao(treino, TK, max_length=64)
    lt = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(123))
    lv = DataLoader(DatasetDeClassificacao(pd.read_csv(PASTA_SPAM / "validacao.csv").head(64), TK, max_length=64),
                    batch_size=8)

    m = adaptar_para_classificacao(_modelo_pequeno(), blocos_treinaveis=None)
    p0, a0 = perda_classificacao_do_loader(lt, m, "cpu"), acuracia_do_loader(lt, m, "cpu")
    h = treinar_classificador(m, lt, lv, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1), "cpu",
                              num_epocas=6, frequencia_avaliacao=10, lotes_avaliacao=None, verboso=False)
    p1, a1 = perda_classificacao_do_loader(lt, m, "cpu"), acuracia_do_loader(lt, m, "cpu")
    passos = 6 * len(lt)
    _verificar(p1 < 0.5 * p0 and a1 > a0,
               f"modelo pequeno treinado do zero: perda {p0:.3f} -> {p1:.3f}, acurácia {a0:.2f} -> {a1:.2f}")
    _verificar(len(h["passos"]) == -(-passos // 10) and len(h["acuracia_validacao"]) == 6 and not m.training,
               f"histórico: {len(h['passos'])} avaliações em {passos} passos, 6 épocas; modelo em eval ao final")

    m2 = adaptar_para_classificacao(_modelo_pequeno(), blocos_treinaveis=1)
    antes = {n: p.clone() for n, p in m2.named_parameters()}
    treinar_classificador(m2, lt, lv, torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=1e-3),
                          "cpu", 1, verboso=False)
    congelados = [n for n, p in m2.named_parameters() if not p.requires_grad]
    atual = dict(m2.named_parameters())
    _verificar(all(torch.equal(antes[n], atual[n]) for n in congelados)
               and any(not torch.equal(antes[n], atual[n]) for n in antes if n not in congelados),
               f"o treino altera só as camadas treináveis ({len(congelados)} tensores congelados intactos)")

    iguais = sum(
        NOMES_DAS_CLASSES.index(classificar_texto(treino["texto"][i], m, TK, max_length=64)[0])
        == int(logits_de_classe(m, ds[i][0].unsqueeze(0)).argmax())
        for i in range(40))
    classe, probs = classificar_texto("oi " * 500, m, TK, max_length=64)
    _verificar(iguais == 40, "classificar_texto dá a mesma classe que o caminho do loader (40 de 40)")
    _verificar(classe in NOMES_DAS_CLASSES and abs(sum(probs.values()) - 1) < 1e-6,
               "texto maior que o contexto é truncado; as probabilidades somam 1")
    _verificar(_levanta(lambda: classificar_texto("", m, TK, max_length=64)), "texto vazio -> ValueError")


# ---------------------------------------------------------------------------
# D e E. Fine-tuning de instruções
# ---------------------------------------------------------------------------
def teste_dados_de_instrucoes() -> tuple[list[dict], list[dict], list[dict]]:
    """Seções 7.2 a 7.4 — formato do prompt, divisão e lotes (D1–D3)."""
    print("\nBase de instruções, prompt e lotes (seções 7.2 a 7.4)")
    bruto = carregar_instrucoes()
    dados = remover_instrucoes_repetidas(bruto)
    treino, validacao, teste = dividir_instrucoes(dados)
    chaves = lambda xs: {(x["instruction"], x["input"]) for x in xs}  # noqa: E731
    _verificar((len(bruto), len(dados)) == (1100, 1075), "1.100 exemplos lidos, 1.075 após remover instruções repetidas")
    _verificar((len(treino), len(validacao), len(teste)) == (913, 55, 107), "divisão 85/5/10: 913 / 55 / 107")
    _verificar(not (chaves(treino) & chaves(validacao) or chaves(treino) & chaves(teste)
                    or chaves(validacao) & chaves(teste)), "nenhuma instrução aparece em mais de um conjunto")

    com = {"instruction": "Identify the correct spelling of the following word.", "input": "Ocassion",
           "output": "The correct spelling is 'Occasion.'"}
    sem = dict(com, input="")
    abertura = ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.")
    corpo = "\n\n### Instruction:\nIdentify the correct spelling of the following word."
    _verificar(formatar_entrada(com) == abertura + corpo + "\n\n### Input:\nOcassion"
               and formatar_entrada(sem) == abertura + corpo,
               "prompt no formato Alpaca; o bloco '### Input' some quando a entrada é vazia")
    _verificar(formatar_exemplo(com) == formatar_entrada(com) + "\n\n### Response:\n" + com["output"]
               and all(formatar_exemplo(x).startswith(montar_prompt(x)) for x in treino)
               and montar_prompt(com).endswith(MARCADOR_RESPOSTA),
               "o prompt de inferência é o prompt de treino cortado logo após '### Response:'")

    p, i = ID_PADDING, INDICE_IGNORADO
    entradas, alvos = agrupar_lote([[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]])
    _verificar(entradas.tolist() == [[0, 1, 2, 3, 4], [5, 6, p, p, p], [7, 8, 9, p, p]]
               and alvos.tolist() == [[1, 2, 3, 4, p], [6, p, i, i, i], [8, 9, p, i, i]],
               "lote do exemplo do livro: padding até o maior item, alvo deslocado, um <|endoftext|>, resto -100")
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 60000)
    validas = alvos.flatten() != i
    manual = -torch.log_softmax(logits.flatten(0, 1), -1)[validas, alvos.flatten()[validas]].mean()
    _verificar(torch.allclose(torch.nn.functional.cross_entropy(logits.flatten(0, 1), alvos.flatten()), manual),
               f"a entropia cruzada considera só as {int(validas.sum())} posições que não são -100")
    _verificar(agrupar_lote([[0, 1, 2, 3, 4], [5, 6]], max_length=3)[0].shape == (2, 3), "max_length limita o lote")

    ds = DatasetDeInstrucoes(treino[:8], TK)
    _verificar(all(TK.decode(ds[k]) == formatar_exemplo(treino[k]) for k in range(8)),
               "cada item do dataset decodifica para o prompt de treino completo")
    lt, lv, lte = criar_loaders_instrucoes(treino, validacao, teste, TK)
    comprimentos = {e.shape[1] for e, _ in lt}
    _verificar((len(lt), len(lv), len(lte)) == (913 // 8, math.ceil(55 / 8), math.ceil(107 / 8))
               and len(comprimentos) > 1,
               f"loaders: {len(lt)} / {len(lv)} / {len(lte)} lotes, com o comprimento definido lote a lote")
    return treino, validacao, teste


def teste_modelo_de_instrucoes(treino: list[dict]) -> None:
    """Seções 7.5 a 7.7 — camadas treináveis, treino e geração da resposta (E1–E2)."""
    print("\nTreino de instruções e geração de respostas (seções 7.5 a 7.7)")
    m = _modelo_pequeno(n_layers=4)
    total = sum(p.numel() for p in m.parameters())
    bloco = sum(p.numel() for p in m.trf_blocks[-1].parameters())
    _verificar(definir_treinaveis(m, 2) == (2 * bloco + 128, total)
               and not m.out_head.weight.requires_grad and not m.embedding.token_emb.weight.requires_grad,
               "blocos_treinaveis=2: dois últimos blocos + norma final; embeddings e cabeça congelados")
    _verificar(definir_treinaveis(m, None) == (total, total), "blocos_treinaveis=None: modelo inteiro")
    _verificar(_levanta(lambda: definir_treinaveis(m, 0)) and _levanta(lambda: definir_treinaveis(m, 5)),
               "número de blocos fora do intervalo -> ValueError")

    exemplos = treino[:4]
    m = _modelo_pequeno(emb_dim=128, context_length=128)
    lt, lv, _ = criar_loaders_instrucoes(exemplos, exemplos, exemplos, TK, batch_size=4, max_length=128)
    e, a = next(iter(lv))
    p0 = perda_do_lote(e, a, m, "cpu").item()
    with contextlib.redirect_stdout(io.StringIO()):
        h = treinar_modelo(m, lt, lv, torch.optim.AdamW(m.parameters(), lr=2e-3), "cpu", num_epocas=150,
                           frequencia_avaliacao=50, lotes_avaliacao=1)
    _verificar(abs(p0 - math.log(V)) < 0.5 and h["perda_treino"][-1] < 0.1,
               f"o laço da Sprint 5 treina com os lotes de instrução: perda {p0:.2f} -> {h['perda_treino'][-1]:.3f}")
    respostas = [responder(x, m, TK, max_novos_tokens=60) for x in exemplos]
    _verificar(respostas == [x["output"] for x in exemplos],
               "modelo que decorou 4 exemplos devolve só a resposta esperada e para sozinho no <|endoftext|>")
    novo = _modelo_pequeno(semente=1, n_layers=1, context_length=128)
    n = len(TK.encode(responder(exemplos[0], novo, TK, max_novos_tokens=12)))
    _verificar(n >= 10, f"modelo sem treino não para: usa o limite de tokens ({n} de 12)")
    _verificar(_levanta(lambda: responder("palavra " * 200, novo, TK)), "prompt maior que o contexto -> ValueError")


def teste_avaliacao_das_respostas() -> None:
    """Seção 7.8 — avaliação das respostas (E4)."""
    print("\nAvaliação das respostas (seção 7.8)")
    a, b = "The capital of France is Paris.", "the capital of france is paris"
    _verificar(normalizar_texto("The correct spelling is 'Occasion.'") == ["the", "correct", "spelling", "is", "occasion"],
               "normalização: minúsculas, sem pontuação")
    _verificar(correspondencia_exata(a, b) and not correspondencia_exata("Paris is the capital of France", b),
               "correspondência exata ignora caixa e pontuação, mas não a ordem")
    _verificar(abs(f1_de_palavras("The cat sat.", "The cat sat on the mat.") - 2 / 3) < 1e-12,
               "F1 de palavras: 3 em comum, precisão 1, revocação 0,5 -> 0,667")
    _verificar(abs(f1_de_palavras("the the the", "the cat") - 0.4) < 1e-12,
               "palavra repetida só conta quantas vezes aparece na esperada: precisão 1/3, revocação 1/2 -> 0,400")
    _verificar(f1_de_palavras("dog", "cat") == 0.0 and f1_de_palavras("", "cat") == 0.0, "sem palavras em comum -> 0")
    r = avaliar_respostas([(a, b), ("The cat sat.", "The cat sat on the mat."), ("dog", "cat")])
    _verificar(r["exemplos"] == 3 and abs(r["correspondencia_exata"] - 1 / 3) < 1e-12
               and abs(r["f1_de_palavras"] - (1 + 2 / 3) / 3) < 1e-12,
               "resumo de três pares: 1/3 de correspondência exata, F1 médio 0,556")
    _verificar(_levanta(lambda: avaliar_respostas([])), "lista vazia -> ValueError")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Validação dos pesos pré-treinados e do fine-tuning — Sprint 6")
    print("=" * 72)

    teste_pesos()
    if "--completo" in sys.argv[1:]:
        teste_pesos_reais()
    teste_dados_de_classificacao()
    teste_modelo_de_classificacao()
    teste_metricas_de_classificacao()
    teste_treino_de_classificacao()
    treino, _, _ = teste_dados_de_instrucoes()
    teste_modelo_de_instrucoes(treino)
    teste_avaliacao_das_respostas()

    aprovados = sum(ok for ok, _ in _resultados)
    print("\n" + "=" * 72)
    print(f"{aprovados}/{len(_resultados)} verificações aprovadas")
    print("=" * 72)
    return 0 if aprovados == len(_resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())

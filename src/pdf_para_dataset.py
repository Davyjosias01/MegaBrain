"""De arquivos PDF a um corpus de treinamento — extensão da Sprint 6.

O pré-treino da Sprint 5 e o da OpenAI usam **texto corrido**: o modelo aprende
a prever o próximo token, e para isso basta texto, sem rótulos. Este módulo
transforma uma pasta de PDFs nesse texto::

    PDF  ->  texto de cada página  ->  limpeza  ->  um documento por PDF
         ->  corpus.txt (documentos separados por <|endoftext|>)
         ->  janelas (entrada, alvo) de ``dados.py``

O corpus resultante serve para **continuar o pré-treino** do GPT-2 no assunto
dos PDFs (``criar_loaders_treino_validacao`` + ``treinar_modelo``): o modelo
passa a escrever no vocabulário e no estilo desses textos.

O que este módulo **não** faz:

- não gera pares pergunta → resposta. Um dataset de instruções exige que
  alguém (ou outro modelo) escreva as perguntas e as respostas; texto corrido
  só serve para o pré-treino;
- não lê PDF escaneado. A extração usa a camada de texto do arquivo
  (biblioteca ``pypdf``); um PDF que é só imagem devolve texto vazio e aparece
  no relatório como ignorado. Ler esses arquivos exigiria OCR.

Uso::

    from pdf_para_dataset import montar_corpus
    relatorio = montar_corpus("pasta/com/pdfs", "dados/corpus.txt")

Ou pela linha de comando, a partir da raiz do repositório::

    uv run python src/pdf_para_dataset.py pasta/com/pdfs dados/corpus.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FIM_DE_TEXTO = "<|endoftext|>"


# ---------------------------------------------------------------------------
# 1. Extração
# ---------------------------------------------------------------------------
def extrair_paginas(caminho: Path | str, max_paginas: int | None = None) -> list[str]:
    """Texto de cada página de um PDF, na ordem.

    Uma página sem camada de texto (imagem escaneada) devolve ``""``. Um
    arquivo protegido por senha, corrompido ou que não é PDF levanta
    ``ValueError`` com o nome do arquivo.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    caminho = Path(caminho)
    try:
        leitor = PdfReader(str(caminho))
        if leitor.is_encrypted and not leitor.decrypt(""):
            raise ValueError(f"{caminho.name}: PDF protegido por senha.")
        paginas = leitor.pages[:max_paginas] if max_paginas else leitor.pages
        return [(pagina.extract_text() or "") for pagina in paginas]
    except (PyPdfError, OSError, KeyError, TypeError, AttributeError) as erro:
        raise ValueError(f"{caminho.name}: não foi possível ler o PDF ({erro}).") from erro


# ---------------------------------------------------------------------------
# 2. Limpeza
# ---------------------------------------------------------------------------
_SO_NUMERO = re.compile(r"^\s*(?:page|página|pág\.?)?\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)
_FIM_DE_FRASE = re.compile(r"[.!?:;…\"”')\]]$")


def remover_linhas_repetidas(paginas: list[str], fracao: float = 0.5) -> list[str]:
    """Remove cabeçalhos e rodapés: linhas que se repetem em muitas páginas.

    Uma linha (ignorando dígitos, para pegar "Capítulo 3 — 41", "… — 42") que
    aparece em mais de ``fracao`` das páginas é considerada cabeçalho ou
    rodapé. Só se aplica a documentos com 4 páginas ou mais.
    """
    if len(paginas) < 4:
        return paginas
    chave = lambda linha: re.sub(r"\d+", "#", linha.strip().lower())  # noqa: E731
    contagem: dict[str, int] = {}
    for pagina in paginas:
        for c in {chave(l) for l in pagina.splitlines() if l.strip()}:
            contagem[c] = contagem.get(c, 0) + 1
    repetidas = {c for c, n in contagem.items() if n > fracao * len(paginas)}
    return ["\n".join(l for l in pagina.splitlines() if chave(l) not in repetidas) for pagina in paginas]


def limpar_texto(texto: str) -> str:
    """Transforma o texto extraído de um PDF em parágrafos de texto corrido.

    O PDF guarda linhas como elas aparecem na página, não frases. A limpeza:

    1. descarta linhas que são só o número da página;
    2. desfaz a hifenização de fim de linha (``treina-`` + ``mento``);
    3. junta as linhas de um mesmo parágrafo — uma linha que não termina em
       pontuação de fim de frase continua na seguinte;
    4. reduz espaços repetidos e separa os parágrafos por uma linha em branco.
    """
    texto = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").replace("­", "")
    paragrafos: list[str] = []
    atual = ""
    for linha in texto.split("\n"):
        linha = re.sub(r"[ \t ]+", " ", linha).strip()
        if not linha:                      # linha em branco fecha o parágrafo
            if atual:
                paragrafos.append(atual)
                atual = ""
            continue
        if _SO_NUMERO.match(linha):
            continue
        if not atual:
            atual = linha
        elif atual.endswith("-") and linha[:1].islower():
            atual = atual[:-1] + linha     # palavra partida no fim da linha
        elif _FIM_DE_FRASE.search(atual) and linha[:1].isupper() and len(atual) < 60:
            paragrafos.append(atual)       # linha curta e completa: título ou item
            atual = linha
        else:
            atual = atual + " " + linha
    if atual:
        paragrafos.append(atual)
    return "\n\n".join(paragrafos)


def extrair_texto_pdf(caminho: Path | str, max_paginas: int | None = None) -> str:
    """Texto limpo de um PDF inteiro."""
    paginas = remover_linhas_repetidas(extrair_paginas(caminho, max_paginas))
    return limpar_texto("\n\n".join(paginas))


# ---------------------------------------------------------------------------
# 3. Corpus
# ---------------------------------------------------------------------------
def listar_pdfs(origem: Path | str | list) -> list[Path]:
    """PDFs de uma pasta (com subpastas), de um arquivo ou de uma lista, em ordem."""
    if isinstance(origem, (list, tuple)):
        return sorted(Path(p) for p in origem)
    origem = Path(origem)
    if origem.is_file():
        return [origem]
    if not origem.is_dir():
        raise FileNotFoundError(f"pasta ou arquivo não encontrado: {origem}")
    return sorted(p for p in origem.rglob("*") if p.suffix.lower() == ".pdf")


def montar_corpus(
    origem: Path | str | list,
    destino: Path | str | None = None,
    min_caracteres: int = 500,
    max_paginas: int | None = None,
    verboso: bool = True,
) -> dict:
    """Extrai, limpa e reúne os PDFs de ``origem`` num único texto.

    Cada PDF vira um documento; os documentos são separados por
    ``<|endoftext|>``, o mesmo separador usado no pré-treino do GPT-2, para o
    modelo não aprender que o fim de um livro continua no começo do outro.

    Um PDF com menos de ``min_caracteres`` de texto (normalmente um arquivo
    escaneado, sem camada de texto) ou que não pôde ser lido é **ignorado** e
    listado no relatório, com o motivo: nenhum arquivo some em silêncio.

    Returns:
        Dicionário com ``texto`` (o corpus), ``documentos`` (nome, páginas e
        caracteres de cada PDF aproveitado), ``ignorados`` (nome e motivo) e
        ``caracteres``. Se ``destino`` for dado, o corpus também é gravado
        nesse arquivo, em UTF-8.
    """
    arquivos = listar_pdfs(origem)
    if not arquivos:
        raise ValueError(f"nenhum PDF encontrado em {origem}.")
    documentos, ignorados, textos = [], [], []
    for arquivo in arquivos:
        try:
            paginas = extrair_paginas(arquivo, max_paginas)
        except ValueError as erro:
            ignorados.append({"arquivo": arquivo.name, "motivo": str(erro)})
            continue
        texto = limpar_texto("\n\n".join(remover_linhas_repetidas(paginas)))
        if len(texto) < min_caracteres:
            ignorados.append({"arquivo": arquivo.name,
                              "motivo": f"só {len(texto)} caracteres de texto em {len(paginas)} páginas "
                                        "(PDF escaneado, sem camada de texto?)"})
            continue
        documentos.append({"arquivo": arquivo.name, "paginas": len(paginas), "caracteres": len(texto)})
        textos.append(texto)
        if verboso:
            print(f"  {arquivo.name[:60]:<60} {len(paginas):>5} páginas {len(texto):>10,} caracteres")
    if not textos:
        raise ValueError("nenhum PDF tinha texto aproveitável: " + "; ".join(i["motivo"] for i in ignorados[:3]))

    corpus = f"\n{FIM_DE_TEXTO}\n".join(textos)
    if destino is not None:
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(corpus, encoding="utf-8", newline="\n")
    if verboso:
        print(f"  {len(documentos)} PDFs aproveitados, {len(ignorados)} ignorados, {len(corpus):,} caracteres")
        for i in ignorados:
            print(f"  ignorado: {i['motivo']}" if i["arquivo"] in i["motivo"] else
                  f"  ignorado: {i['arquivo']}: {i['motivo']}")
    return {"texto": corpus, "documentos": documentos, "ignorados": ignorados, "caracteres": len(corpus)}


def resumo_do_corpus(texto: str, tokenizador=None, max_length: int = 256) -> dict:
    """Tamanho do corpus em caracteres, palavras, tokens e janelas de treino."""
    if tokenizador is None:
        from tokenizacao import TokenizadorBPE

        tokenizador = TokenizadorBPE()
    tokens = len(tokenizador.encode(texto))
    return {
        "documentos": texto.count(FIM_DE_TEXTO) + 1,
        "caracteres": len(texto),
        "palavras": len(texto.split()),
        "tokens": tokens,
        "tokens_por_palavra": tokens / max(1, len(texto.split())),
        "janelas_de_treino": max(0, (tokens - 1) // max_length),
    }


def main(argumentos: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(argumentos) != 2:
        print("uso: uv run python src/pdf_para_dataset.py <pasta-ou-pdf> <corpus.txt>")
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    relatorio = montar_corpus(argumentos[0], argumentos[1])
    for chave, valor in resumo_do_corpus(relatorio["texto"]).items():
        print(f"  {chave:<20}: {valor:,.2f}" if isinstance(valor, float) else f"  {chave:<20}: {valor:,}")
    print(f"  corpus gravado em {argumentos[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

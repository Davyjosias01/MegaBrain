"""Gera ``glossario-consolidado.md`` a partir dos glossários de cada capítulo.

O glossário consolidado é um **índice**: todos os termos dos arquivos
``capitulo-NN.md`` em ordem alfabética, cada um com o capítulo em que foi
definido, a primeira frase da definição e o link para a entrada completa. As
definições continuam escritas num lugar só, o arquivo do capítulo; este script
é executado de novo sempre que um deles muda.

Um termo que aparece em mais de um capítulo (por exemplo, *fine-tuning*,
apresentado no Capítulo 1 e implementado no 6) ganha uma linha por capítulo.

Uso, a partir da raiz do repositório::

    uv run python technical-glossary/consolidar.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

PASTA = Path(__file__).resolve().parent
DESTINO = PASTA / "glossario-consolidado.md"

CABECALHO = """# Glossário consolidado — Capítulos 1 a 7

Índice de todos os termos técnicos do projeto, em ordem alfabética. Cada linha traz o termo, o
capítulo em que ele foi definido, a primeira frase da definição e o link para a entrada completa — que
tem também a **função no modelo**, a **relação com outros conceitos** e um **exemplo**.

Arquivo gerado por [`consolidar.py`](consolidar.py) a partir dos glossários de cada capítulo; não edite
à mão.

{resumo}

Um termo listado em dois capítulos foi apresentado em nível conceitual no primeiro e retomado, com a
implementação do projeto, no segundo.
"""


def ancora(titulo: str) -> str:
    """Âncora que o GitHub gera para um título de seção."""
    texto = titulo.strip().lower()
    texto = "".join(c for c in texto if c.isalnum() or c in " -_")
    return texto.replace(" ", "-")


def primeira_frase(paragrafo: str) -> str:
    """Primeira frase da definição, numa linha só."""
    texto = " ".join(paragrafo.split())
    texto = re.sub(r"^\*\*Definição\.\*\*\s*", "", texto)
    partes = re.split(r"(?<=[^\s.][^\s.]\.)\s+(?=[A-ZÀ-Ú`*\[])", texto, maxsplit=1)
    return partes[0].replace("|", "\\|")


def entradas_do_capitulo(caminho: Path) -> list[dict]:
    capitulo = int(re.search(r"capitulo-(\d+)", caminho.name).group(1))
    blocos = re.split(r"^## ", caminho.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]
    entradas = []
    for bloco in blocos:
        titulo, _, corpo = bloco.partition("\n")
        definicao = re.search(r"^\*\*Definição\.\*\*.*?(?=\n\s*\n)", corpo, flags=re.MULTILINE | re.DOTALL)
        if definicao is None:
            raise ValueError(f"{caminho.name}: a entrada {titulo!r} não tem '**Definição.**'.")
        entradas.append({
            "termo": titulo.strip(),
            "capitulo": capitulo,
            "link": f"{caminho.name}#{ancora(titulo)}",
            # Links internos do capítulo passam a apontar para o arquivo dele.
            "definicao": primeira_frase(definicao.group(0)).replace("](#", f"]({caminho.name}#"),
        })
    return entradas


def chave_de_ordenacao(termo: str) -> str:
    """Ordem alfabética sem acentos, caixa nem marcação."""
    sem_acento = unicodedata.normalize("NFKD", termo).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", sem_acento.lower()).strip()


def consolidar() -> tuple[str, int, dict[int, int]]:
    arquivos = sorted(PASTA.glob("capitulo-[0-9][0-9].md"))
    entradas = [e for arquivo in arquivos for e in entradas_do_capitulo(arquivo)]
    entradas.sort(key=lambda e: (chave_de_ordenacao(e["termo"]), e["capitulo"]))
    por_capitulo = {}
    for e in entradas:
        por_capitulo[e["capitulo"]] = por_capitulo.get(e["capitulo"], 0) + 1

    resumo = ["| Capítulo | Entradas | Arquivo |", "|---|---|---|"]
    for capitulo, quantidade in sorted(por_capitulo.items()):
        resumo.append(f"| {capitulo} | {quantidade} | [`capitulo-{capitulo:02d}.md`](capitulo-{capitulo:02d}.md) |")
    resumo.append(f"| **Total** | **{len(entradas)}** | |")

    linhas = [CABECALHO.format(resumo="\n".join(resumo))]
    letra_atual = ""
    for e in entradas:
        letra = chave_de_ordenacao(e["termo"])[0].upper()
        if letra != letra_atual:
            letra_atual = letra
            linhas += [f"\n## {letra}\n", "| Termo | Cap. | Definição |", "|---|---|---|"]
        linhas.append(f"| [{e['termo']}]({e['link']}) | {e['capitulo']} | {e['definicao']} |")
    return "\n".join(linhas) + "\n", len(entradas), por_capitulo


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    texto, total, por_capitulo = consolidar()
    DESTINO.write_text(texto, encoding="utf-8", newline="\n")
    print(f"{DESTINO.name}: {total} entradas de {len(por_capitulo)} capítulos {por_capitulo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

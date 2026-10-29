# `src/` — Código-fonte

Implementação dos componentes do LLM. É o núcleo reutilizável do projeto: cada sprint adiciona módulos
aqui, e as sprints seguintes importam o que já foi construído em vez de reescrever.

O código deste diretório deve ser importável e executável de forma independente — notebooks e
experimentos consomem estes módulos, nunca o contrário.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `verificar_ambiente.py` | Valida a instalação do ambiente: versão do Python, PyTorch, disponibilidade de CUDA, operações com tensores e *autograd* |
| `tokenizacao.py` | **Sprint 2** — `dividir_em_tokens`, `construir_vocabulario`, `TokenizadorSimples` (vocabulário próprio, `<\|unk\|>` e `<\|endoftext\|>`) e `TokenizadorBPE` (GPT-2, via tiktoken), com a mesma interface `encode`/`decode` |
| `dados.py` | **Sprint 2** — corpus, `pares_entrada_alvo`, `GPTDatasetV1` (janela deslizante), `numero_de_amostras` e `criar_dataloader` |
| `embeddings.py` | **Sprint 2** — `EmbeddingDeEntrada`: embedding de token + embedding posicional, a primeira camada do modelo GPT |
| `validar_dados.py` | **Sprint 2** — 38 verificações do pipeline de dados contra os valores do Capítulo 2 e contra suas propriedades |
| `attention.py` | **Sprint 3** — mecanismos de atenção: `atencao_simples`, `scaled_dot_product_attention`, `SelfAttentionV1`, `SelfAttentionV2`, `CausalAttention`, `MultiHeadAttentionWrapper` e `MultiHeadAttention` |
| `entradas_atencao.py` | **Sprint 3** — ponte com a Sprint 2: encadeia `tokenizacao.py`, `dados.py` e `embeddings.py` para produzir os lotes de embeddings que alimentam os mecanismos de atenção |
| `validar_atencao.py` | **Sprint 3** — 28 verificações de `attention.py` contra os valores publicados no Capítulo 3 e contra as propriedades matemáticas dos mecanismos |
| `gpt.py` | **Sprint 4** — `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock` e `GPTModel` (reaproveita `EmbeddingDeEntrada` e `MultiHeadAttention`); configurações do GPT-2 (`CONFIG_GPT2_124M`, `config_gpt2`) e contagem de parâmetros e memória |
| `geracao.py` | **Sprint 4** — inferência e geração autorregressiva: `texto_para_ids`, `ids_para_texto`, `selecionar_proximo_token` (greedy, temperatura, top-k), `gerar` e `gerar_texto_simples` |
| `validar_gpt.py` | **Sprint 4** — 38 verificações da arquitetura e da geração contra os valores do Capítulo 4 e da seção 5.3 |

## Organização prevista

Conforme as sprints avançam, o diretório receberá os módulos correspondentes a cada etapa do pipeline:

- **Sprint 2** *(concluída)* — tokenização, vocabulário, dataset e *dataloader*, *embeddings* e *positional embeddings*
- **Sprint 3** *(concluída)* — mecanismos de atenção: *self-attention*, *scaled dot-product*, *causal* e *multi-head*
- **Sprint 4** *(concluída)* — arquitetura GPT: *layer normalization*, *feed forward network*, conexões residuais e blocos Transformer
- **Sprint 5** — laço de treinamento, função de perda, otimizadores e geração de texto
- **Sprint 6** — estratégias de *fine-tuning*

## Execução

Com o ambiente virtual ativo, a partir da raiz do repositório:

```bash
python src/verificar_ambiente.py   # valida o ambiente (Sprint 0)
python src/validar_dados.py        # valida o pipeline de dados (Sprint 2)
python src/validar_atencao.py      # valida os mecanismos de atenção (Sprint 3)
python src/validar_gpt.py          # valida a arquitetura GPT e a geração (Sprint 4)
```

# `src/` — Código-fonte

Implementação dos componentes do LLM. É o núcleo reutilizável do projeto: cada sprint adiciona módulos
aqui, e as sprints seguintes importam o que já foi construído em vez de reescrever.

O código deste diretório deve ser importável e executável de forma independente — notebooks e
experimentos consomem estes módulos, nunca o contrário.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `verificar_ambiente.py` | Valida a instalação do ambiente: versão do Python, PyTorch, disponibilidade de CUDA, operações com tensores e *autograd* |
| `attention.py` | **Sprint 3** — mecanismos de atenção: `atencao_simples`, `scaled_dot_product_attention`, `SelfAttentionV1`, `SelfAttentionV2`, `CausalAttention`, `MultiHeadAttentionWrapper` e `MultiHeadAttention` |
| `entradas_atencao.py` | **Sprint 3** — ponte com a Sprint 2: tokenização BPE, janela deslizante e embeddings (de token e posicionais) que alimentam os mecanismos de atenção |
| `validar_atencao.py` | **Sprint 3** — 28 verificações de `attention.py` contra os valores publicados no Capítulo 3 e contra as propriedades matemáticas dos mecanismos |

## Organização prevista

Conforme as sprints avançam, o diretório receberá os módulos correspondentes a cada etapa do pipeline:

- **Sprint 2** — tokenização, vocabulário, dataset e *dataloader*, *embeddings* e *positional embeddings*
- **Sprint 3** — mecanismos de atenção: *self-attention*, *scaled dot-product*, *causal* e *multi-head*
- **Sprint 4** — arquitetura GPT: *layer normalization*, *feed forward network*, conexões residuais e blocos Transformer
- **Sprint 5** — laço de treinamento, função de perda, otimizadores e geração de texto
- **Sprint 6** — estratégias de *fine-tuning*

## Execução

Com o ambiente virtual ativo, a partir da raiz do repositório:

```bash
python src/verificar_ambiente.py   # valida o ambiente (Sprint 0)
python src/validar_atencao.py      # valida os mecanismos de atenção (Sprint 3)
```

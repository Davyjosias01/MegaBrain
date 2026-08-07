# `src/` — Código-fonte

Implementação dos componentes do LLM. É o núcleo reutilizável do projeto: cada sprint adiciona módulos
aqui, e as sprints seguintes importam o que já foi construído em vez de reescrever.

O código deste diretório deve ser importável e executável de forma independente — notebooks e
experimentos consomem estes módulos, nunca o contrário.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `verificar_ambiente.py` | Valida a instalação do ambiente: versão do Python, PyTorch, disponibilidade de CUDA, operações com tensores e *autograd* |

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
python src/verificar_ambiente.py
```

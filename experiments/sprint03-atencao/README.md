# `sprint03-atencao` — Experimentos dos mecanismos de atenção

Experimentos da Sprint 3, sobre os mecanismos implementados em
[`src/attention.py`](../../src/attention.py). Cada script varia **uma** dimensão do problema e mede o
efeito sobre o custo computacional e sobre o formato da distribuição de atenção.

A interpretação dos resultados está em [`reports/sprint03-relatorio.md`](../../reports/sprint03-relatorio.md).

## Execução

```bash
cd experiments/sprint03-atencao
python executar_todos.py
```

Ou um experimento isolado:

```bash
python exp06_com_e_sem_escala.py
```

Todos os artefatos vão para [`results-by-sprints/sprint03/`](../../results-by-sprints/sprint03/) —
13 figuras `.png`, 10 tabelas `.csv` e o registro completo do console em `execucao.log`, que inclui o
ambiente de execução (versão do PyTorch, número de threads, processador).

## Os experimentos

| Script | O que varia | O que mede |
|---|---|---|
| `exp01_dimensao_embedding.py` | `d_in = d_out` ∈ {16 … 512} | parâmetros, tempo, entropia da atenção |
| `exp02_numero_de_heads.py` | `num_heads` ∈ {1 … 16}, `d_out` fixo | parâmetros, tempo, especialização das cabeças |
| `exp03_dimensao_de_head.py` | `head_dim` ∈ {4 … 128}, `num_heads` fixo | parâmetros, tempo, entropia |
| `exp04_self_vs_multihead.py` | o mecanismo | custo dos cinco mecanismos; empilhamento × divisão de pesos |
| `exp05_visualizacao_matriz.py` | — | mapas de calor das matrizes de atenção |
| `exp06_com_e_sem_escala.py` | presença da escala `1/√d_k`, `d_k` ∈ {2 … 2048} | saturação do softmax e gradiente |
| `exp07_mascara_causal.py` | máscara ligada/desligada, taxa de dropout | causalidade, vazamento de informação, dropout |
| `exp08_sequencias_de_entrada.py` | conteúdo e comprimento da sequência | distribuição da atenção, custo quadrático |

`comum.py` concentra o que é compartilhado: semente, caminhos, estilo dos gráficos, medição de tempo
e as métricas sobre a matriz de atenção. `executar_todos.py` roda os oito na ordem e registra o
ambiente de execução no log.

## Métricas

**Entropia normalizada** — entropia de Shannon das linhas da matriz de atenção, dividida por `log(k)`,
onde `k` é o número de posições visíveis naquela linha. Fica em `[0, 1]`: perto de 1 a atenção é
quase uniforme, perto de 0 está concentrada em um único token. A normalização torna o valor
comparável entre sequências e configurações de tamanhos diferentes. Em atenção causal a linha 0 é
ignorada, pois tem uma única posição visível e peso sempre 1.

**Dissimilaridade entre cabeças** — distância L1 média entre as matrizes de atenção de cada par de
cabeças, normalizada pelo número de tokens. Zero significaria cabeças redundantes.

**Sensibilidade do softmax** — traço da jacobiana do softmax, `Σⱼ pⱼ(1 − pⱼ)`. Mede diretamente
quanto gradiente o softmax deixa passar, sem depender da função de perda escolhida.

## Reprodutibilidade

- Semente fixa em 123, reaplicada antes de instanciar cada módulo medido.
- Execução em CPU, com o número de threads registrado no log.
- Entradas reais vindas de `sprints/the-verdict.txt`, pelo pipeline da Sprint 2
  ([`src/entradas_atencao.py`](../../src/entradas_atencao.py)).
- **Nenhum módulo é treinado.** Todas as medidas são feitas com pesos na inicialização — os
  experimentos caracterizam o comportamento do *mecanismo*, não padrões linguísticos aprendidos.
- As medidas de tempo dependem da máquina e variam entre execuções; devem ser lidas como ordens de
  grandeza e comparações relativas, não como valores absolutos.

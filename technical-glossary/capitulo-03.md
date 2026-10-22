# Capítulo 3 — Coding Attention Mechanisms

Glossário dos termos introduzidos no Capítulo 3 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*. Entradas em ordem alfabética.

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `x⁽ⁱ⁾` | embedding do *i*-ésimo token da sequência |
| `z⁽ⁱ⁾` | vetor de contexto do *i*-ésimo token |
| `ω` (ômega) | score de atenção, antes da normalização |
| `α` (alfa) | peso de atenção, depois do softmax |
| `Q`, `K`, `V` | matrizes de queries, keys e values |
| `Wq`, `Wk`, `Wv` | matrizes de pesos treináveis que produzem `Q`, `K` e `V` |
| `d_in` / `d_out` | dimensão do embedding de entrada / do vetor de contexto |
| `d_k` | dimensão das keys (igual a `head_dim` em multi-head) |
| `T` ou `n` | número de tokens da sequência |

---

## Attention score (score de atenção)

**Definição.** Valor não normalizado que mede o quanto um token deve prestar atenção em outro.
É o produto escalar entre a query de um token e a key de outro: `ωᵢⱼ = qᵢ · kⱼ`. Na versão
simplificada da seção 3.3, sem pesos treináveis, é o produto escalar dos próprios embeddings:
`ωᵢⱼ = x⁽ⁱ⁾ · x⁽ʲ⁾`.

**Função no modelo.** É a matéria-prima da atenção. O produto escalar mede alinhamento entre
vetores — quanto maior, mais parecidas as direções —, e o mecanismo usa essa similaridade como
critério de relevância entre tokens.

**Relação com outros conceitos.** Vira [peso de atenção](#attention-weight-peso-de-atenção) depois
da [escala](#scaling-factor-fator-de-escala-1d_k) e do [softmax](#softmax). Não confundir com
[peso do modelo](#weight-parameter-parâmetro-de-peso).

**Exemplo.** Para as seis palavras de *"Your journey starts with one step"*, a matriz completa de
scores é `inputs @ inputs.T` — uma matriz `6 × 6`.

---

## Attention weight (peso de atenção)

**Definição.** Score de atenção normalizado por softmax, de modo que os pesos de uma mesma linha
somem 1 e sejam todos positivos: `αᵢⱼ = softmax(ωᵢ)ⱼ`.

**Função no modelo.** Define a fração com que cada token entra no vetor de contexto de outro. Por
somarem 1, os pesos podem ser lidos como uma distribuição de importância relativa.

**Relação com outros conceitos.** São a matriz visualizada nos mapas de calor. São dinâmicos —
mudam a cada entrada —, ao contrário dos [parâmetros de peso](#weight-parameter-parâmetro-de-peso),
que são fixos depois do treinamento.

**Exemplo.** `attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)`.

---

## Bahdanau attention

**Definição.** Mecanismo de atenção proposto em 2014 para redes recorrentes, que permite ao decoder
acessar seletivamente todos os estados ocultos do encoder, em vez de depender apenas do último.

**Função no modelo.** Não é usado neste projeto. Importa como antecedente histórico: foi a ideia de
acesso seletivo que inspirou a self-attention do Transformer, três anos depois.

**Relação com outros conceitos.** Atua entre *duas* sequências (entrada e saída); a
[self-attention](#self-attention-autoatenção) atua dentro de *uma*.

**Exemplo.** Na tradução de uma frase do alemão para o inglês (figuras 3.3 a 3.5), a cada palavra
inglesa gerada o decoder calcula pesos sobre todos os estados ocultos do encoder — um por palavra
alemã — e pode se concentrar nas palavras de origem que correspondem à que está sendo gerada, mesmo
quando a ordem das palavras difere entre os dois idiomas.

---

## Causal attention (atenção causal) / Masked attention

**Definição.** Variante da self-attention em que cada token só pode atender a si mesmo e às posições
anteriores. Implementada zerando os pesos acima da diagonal da matriz de atenção.

**Função no modelo.** Torna o modelo autorregressivo. Um LLM é treinado para prever o próximo token;
se pudesse ler as posições futuras ao computar o contexto de uma posição, aprenderia a copiar a
resposta em vez de prevê-la, e falharia na geração, quando o futuro ainda não existe.

**Relação com outros conceitos.** Usa a [máscara causal](#causal-mask-máscara-causal); é a base da
[multi-head attention](#multi-head-attention-atenção-multi-cabeça) do GPT.

**Exemplo.** Em `src/attention.py`, a classe `CausalAttention`.

---

## Causal mask (máscara causal)

**Definição.** Matriz triangular que marca as posições a esconder — o triângulo estritamente acima da
diagonal. Aplicada de duas formas equivalentes:

1. multiplicar os pesos por `torch.tril(...)` e renormalizar cada linha (figura 3.20);
2. preencher os scores com `-inf` antes do softmax, via `masked_fill` (figura 3.21).

**Função no modelo.** Implementa a restrição causal. A segunda forma é preferida: como `e^{-∞} = 0`,
o softmax já atribui peso zero às posições escondidas e renormaliza as restantes em um único passo.

**Relação com outros conceitos.** No código é guardada com [`register_buffer`](#register_buffer),
por ser estado do módulo sem ser parâmetro treinável.

**Exemplo.** `torch.triu(torch.ones(n, n), diagonal=1)`.

---

## Context vector (vetor de contexto)

**Definição.** Saída da atenção para um token: a soma dos vetores de value de toda a sequência,
ponderada pelos pesos de atenção — `z⁽ⁱ⁾ = Σⱼ αᵢⱼ · vⱼ`. Na versão sem pesos treináveis, a soma é
sobre os próprios embeddings de entrada.

**Função no modelo.** É o *embedding enriquecido* do token: parte da representação isolada do token
e incorpora informação das outras posições relevantes. É o que permite que a mesma palavra tenha
representações diferentes em frases diferentes.

**Relação com outros conceitos.** Substitui o embedding de entrada na entrada da camada seguinte do
[bloco Transformer](#transformer-block-bloco-transformer).

**Exemplo.** Na atenção simplificada, o vetor de contexto de *"journey"* é
`z⁽²⁾ = [0.4419, 0.6515, 0.5683]`: a combinação dos seis embeddings de entrada com os pesos
`[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`. Com pesos treináveis (`SelfAttentionV1`, semente
123), passa a ser `[0.3061, 0.8210]`, com dimensão `d_out = 2`.

---

## Dot product (produto escalar)

**Definição.** Soma dos produtos elemento a elemento de dois vetores: `a · b = Σᵢ aᵢbᵢ`.

**Função no modelo.** É a medida de similaridade que a atenção usa. Vetores apontando para a mesma
direção produzem produto escalar alto; vetores ortogonais, próximo de zero.

**Relação com outros conceitos.** Calculado em massa por multiplicação de matrizes: `Q @ K.T`
substitui os laços aninhados sobre pares de tokens, com o mesmo resultado e muito mais rápido.

**Exemplo.** `x⁽¹⁾ · x⁽²⁾ = 0.43·0.55 + 0.15·0.87 + 0.89·0.66 = 0.9544` — o score de *"Your"* na linha
da query *"journey"* (p. 72). Em código, `torch.dot(inputs[0], inputs[1])`.

---

## Dropout

**Definição.** Técnica de regularização que zera aleatoriamente uma fração `p` dos elementos de um
tensor durante o treinamento e multiplica os sobreviventes por `1/(1-p)`.

**Função no modelo.** No Capítulo 3 é aplicado sobre os pesos de atenção já normalizados, impedindo
que o modelo fique dependente de um conjunto fixo de ligações entre tokens. O reescalonamento
preserva a *esperança* da saída, de modo que treino e inferência sejam comparáveis — mas quebra a
soma 1 de cada linha individual.

**Relação com outros conceitos.** Só age em modo de treino; `module.eval()` o desativa. O livro usa
0.5 nos exemplos e 0.1 ou 0.2 no treinamento.

**Exemplo.** Com `torch.manual_seed(123)`, `nn.Dropout(0.5)` aplicado a `torch.ones(6, 6)` zera cerca
de metade dos elementos e transforma os demais em 2 (`1 / (1 − 0.5)`); a primeira linha fica
`[2, 2, 0, 2, 2, 0]`.

---

## Head dimension (`head_dim`)

**Definição.** Dimensão do subespaço em que cada cabeça compara queries e keys:
`head_dim = d_out // num_heads`.

**Função no modelo.** É o `d_k` da fórmula da atenção — aparece tanto no produto escalar quanto no
divisor da escala. Define quanta capacidade de representação cada cabeça recebe do orçamento total.

**Relação com outros conceitos.** No GPT-2 de 117 milhões de parâmetros, `d_out = 768` e
`num_heads = 12`, logo `head_dim = 64`.

**Exemplo.** No experimento 02 da Sprint 3, com `d_out = 96` fixo, variar `num_heads` de 1 a 16 leva
`head_dim` de 96 a 6, e a contagem de parâmetros fica travada em 36.960: muda só a forma como as
colunas de `Wq`, `Wk` e `Wv` são repartidas entre as cabeças.

---

## Information leakage (vazamento de informação)

**Definição.** Situação em que a representação de um token incorpora informação de posições que
deveriam estar escondidas.

**Função no modelo.** É exatamente o que a máscara causal previne. O livro observa que mascarar
*depois* do softmax e renormalizar não causa vazamento: renormalizar equivale a recalcular o softmax
apenas sobre as posições visíveis, e a contribuição das posições mascaradas é anulada.

**Relação com outros conceitos.** É o defeito que a [máscara causal](#causal-mask-máscara-causal)
previne e o que torna a [atenção causal](#causal-attention-atenção-causal--masked-attention)
necessária. Está ligado ao formato dos
[pares entrada-alvo](capitulo-02.md#input-target-pairs-pares-entrada-alvo) da Sprint 2: como o alvo da
posição *i* é o token *i + 1*, qualquer vazamento do futuro entrega a resposta ao modelo.

**Exemplo.** Teste prático: perturbar o embedding de um token e verificar que nenhum vetor de
contexto anterior a ele se altera (experimento 07 da Sprint 3).

---

## Key (chave)

**Definição.** Projeção do embedding de entrada por `Wk`: `kᵢ = x⁽ⁱ⁾ · Wk`.

**Função no modelo.** É o "rótulo" que cada token oferece para ser encontrado. O score de atenção
nasce da comparação entre a query de um token e as keys de todos os outros.

**Relação com outros conceitos.** A analogia é com bancos de dados: a *query* é a busca, a *key* é o
índice, o *value* é o conteúdo devolvido.

**Exemplo.** Com `SelfAttentionV1(d_in=3, d_out=2)` e semente 123, a key de *"journey"* é
`k⁽²⁾ = x⁽²⁾ · Wk = [0.4433, 1.1419]`; as keys das seis palavras formam uma matriz `6 × 2`
(`keys.shape == (6, 2)`).

---

## Multi-head attention (atenção multi-cabeça)

**Definição.** Execução de vários mecanismos de atenção em paralelo sobre a mesma sequência, cada um
com suas próprias matrizes `Wq`, `Wk` e `Wv`, com as saídas concatenadas.

**Função no modelo.** Cada cabeça compara tokens em um subespaço diferente e pode se especializar em
um tipo de relação. Uma única cabeça precisaria comprimir todos os tipos de dependência em uma só
distribuição de atenção.

**Relação com outros conceitos.** O livro apresenta duas implementações:
[empilhamento](#multiheadattentionwrapper-empilhamento-de-cabeças) e
[divisão de pesos](#multiheadattention-divisão-de-pesos).

**Exemplo.** No GPT-2 pequeno, 12 cabeças com `head_dim = 64` somam `d_out = 768`. Uma
`MultiHeadAttention(768, 768, 1024, 0.0, 12)` tem 2.360.064 parâmetros (exercício 3.3) e devolve
pesos de atenção de forma `(b, 12, n, n)` — uma matriz `n × n` por cabeça.

---

## `MultiHeadAttentionWrapper` (empilhamento de cabeças)

**Definição.** Implementação didática (listagem 3.4): uma `nn.ModuleList` de `num_heads` módulos
`CausalAttention` independentes, com as saídas concatenadas na última dimensão.

**Função no modelo.** Torna o conceito explícito — cada cabeça é literalmente um objeto separado.

**Relação com outros conceitos.** A dimensão de saída é `d_out * num_heads`, e as cabeças são
executadas em laço Python, uma após a outra.

**Exemplo.** Com `d_in = 3`, `d_out = 2` e `num_heads = 2`, um lote `(2, 6, 3)` com duas cópias da
frase *"Your journey starts with one step"* produz saída `(2, 6, 4)`: duas cabeças de 2 dimensões,
concatenadas (p. 103).

---

## `MultiHeadAttention` (divisão de pesos)

**Definição.** Implementação eficiente (listagem 3.5): uma única projeção linear de tamanho `d_out`
por papel, cujo resultado é fatiado em `num_heads` blocos de `head_dim` colunas por `view` e
`transpose`.

**Função no modelo.** Matematicamente idêntica ao empilhamento, mas substitui `3 × num_heads`
multiplicações de matriz por 3. É a classe que será integrada ao bloco Transformer na Sprint 4.

**Relação com outros conceitos.** Acrescenta a [projeção de saída](#output-projection-out_proj),
ausente no wrapper. Aqui `d_out` é a dimensão **total** da saída, não a de cada cabeça.

**Exemplo.** Caminho dos tensores:
`(b, n, d_in) → (b, n, d_out) → (b, n, num_heads, head_dim) → (b, num_heads, n, head_dim)`.

---

## Output projection (`out_proj`)

**Definição.** Camada `nn.Linear(d_out, d_out)` aplicada depois de reunir as saídas das cabeças.

**Função no modelo.** Mistura as informações produzidas pelas cabeças, que até esse ponto estão
apenas justapostas. Não é estritamente necessária, mas é padrão nas arquiteturas de LLM.

**Relação com outros conceitos.** É a diferença de parâmetros entre `MultiHeadAttention` e
`MultiHeadAttentionWrapper`.

**Exemplo.** Em uma `MultiHeadAttention` com `d_out = 768`, a `out_proj` tem
`768 × 768 + 768 = 590.592` parâmetros (pesos mais bias) — cerca de um quarto dos 2.360.064 do módulo.

---

## Query (consulta)

**Definição.** Projeção do embedding de entrada por `Wq`: `qᵢ = x⁽ⁱ⁾ · Wq`.

**Função no modelo.** Representa o que o token *procura* no restante da sequência. Cada linha da
matriz de atenção corresponde a uma query.

**Relação com outros conceitos.** Ter `Wq` e `Wk` distintos é o que permite relações assimétricas —
o token *A* pode atender fortemente *B* sem a recíproca. Na atenção simplificada da seção 3.3, sem
essas matrizes, a matriz de scores é necessariamente simétrica.

**Exemplo.** Com `SelfAttentionV1(d_in=3, d_out=2)` e semente 123, a query de *"journey"* é
`q⁽²⁾ = x⁽²⁾ · Wq = [0.4306, 1.4551]`; comparada com as seis keys, produz a segunda linha da matriz de
scores.

---

## `register_buffer`

**Definição.** Método do `nn.Module` que registra um tensor como estado do módulo sem torná-lo
parâmetro treinável.

**Função no modelo.** Usado para a máscara causal. O tensor acompanha o módulo em `.to(device)` e
aparece no `state_dict`, mas não recebe gradiente nem é atualizado pelo otimizador — evitando erros
de dispositivo quando o modelo vai para a GPU.

**Relação com outros conceitos.** Contrapõe-se ao `nn.Parameter` dos
[parâmetros de peso](#weight-parameter-parâmetro-de-peso): parâmetros aparecem em
`module.parameters()` e são atualizados pelo otimizador; buffers aparecem apenas no `state_dict` e em
`named_buffers()`. Guarda a [máscara causal](#causal-mask-máscara-causal) de `CausalAttention` e
`MultiHeadAttention`, criada com tamanho `context_length` e recortada para `[:n, :n]` no `forward`.

**Exemplo.** Em `CausalAttention(3, 2, 6, 0.0)`, `named_parameters()` lista apenas `W_query.weight`,
`W_key.weight` e `W_value.weight`, enquanto `state_dict()` lista também `mask` — um tensor booleano
`6 × 6`.

---

## Scaled dot-product attention (atenção por produto escalar escalado)

**Definição.** A self-attention com pesos treináveis usada no Transformer e nos modelos GPT:

```
Atenção(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

**Função no modelo.** É o mecanismo central do Capítulo 3. O nome vem do
[fator de escala](#scaling-factor-fator-de-escala-1d_k) `1/√d_k`.

**Relação com outros conceitos.** Com [máscara causal](#causal-mask-máscara-causal) torna-se
`CausalAttention`; replicada em paralelo, torna-se multi-head attention.

**Exemplo.** Em `src/attention.py`, a função `scaled_dot_product_attention(queries, keys, values)` é o
núcleo compartilhado por todas as classes com pesos treináveis. Com `SelfAttentionV2(3, 2)` e semente
789, a primeira linha da saída para *"Your journey starts with one step"* é `[-0.0739, 0.0713]`
(p. 89).

---

## Scaling factor (fator de escala, `1/√d_k`)

**Definição.** Divisão dos scores de atenção pela raiz quadrada da dimensão das keys, antes do
softmax.

**Função no modelo.** Evita gradientes minúsculos. O produto escalar de dois vetores de dimensão
`d_k` tem desvio-padrão proporcional a `√d_k`; sem a divisão, scores grandes fazem o softmax se
comportar como uma função degrau, sua derivada tende a zero e o treinamento estagna. Dividir por
`√d_k` mantém o desvio-padrão dos scores próximo de 1, qualquer que seja a dimensão.

**Relação com outros conceitos.** Dá nome à
[scaled dot-product attention](#scaled-dot-product-attention-atenção-por-produto-escalar-escalado);
age sobre os [scores de atenção](#attention-score-score-de-atenção) antes do [softmax](#softmax),
evitando sua saturação. Em multi-head, o `d_k` é o [`head_dim`](#head-dimension-head_dim), e não
`d_out`: com 12 cabeças de 64 dimensões, o divisor é `√64 = 8`.

**Exemplo.** No experimento 06 da Sprint 3, com `d_k = 2048` e sem escala, o peso máximo médio por
linha chega a 0,97 e a sensibilidade do softmax cai de 0,96 para 0,04.

---

## Self-attention (autoatenção)

**Definição.** Mecanismo em que cada posição de uma sequência atende a todas as posições *da mesma
sequência* para construir sua representação.

**Função no modelo.** Substitui a recorrência das RNNs: em vez de comprimir a sequência inteira em um
único estado oculto passado adiante, todas as posições ficam simultaneamente acessíveis, e as
dependências de longa distância não se perdem.

**Relação com outros conceitos.** O "self" opõe-se à atenção entre duas sequências diferentes, como
na [Bahdanau attention](#bahdanau-attention). O livro a apresenta em dois estágios: sem pesos
treináveis (seção 3.3) e com pesos treináveis (seção 3.4).

**Exemplo.** Em *"Your journey starts with one step"*, a linha de *"journey"* na matriz de pesos da
atenção simplificada é `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`: o token atende mais a si
mesmo (0,2379) e a *"starts"* (0,2333), cujo embedding é quase paralelo ao seu.

---

## `SelfAttentionV1` e `SelfAttentionV2`

**Definição.** Duas implementações da mesma self-attention: a primeira declara `Wq`, `Wk` e `Wv` como
`nn.Parameter(torch.rand(...))`; a segunda usa `nn.Linear(..., bias=False)`.

**Função no modelo.** São matematicamente equivalentes — uma camada linear sem bias *é* uma
multiplicação de matrizes. A diferença está na inicialização: `nn.Linear` usa um esquema que
controla a variância das ativações, o que torna o treinamento mais estável. Por isso produzem
números diferentes a partir da mesma semente.

**Relação com outros conceitos.** `nn.Linear` guarda a matriz de pesos **transposta**, com forma
`(d_out, d_in)`; transferir pesos de uma versão para a outra exige `W.weight.T` (exercício 3.1).

**Exemplo.** As duas têm 18 parâmetros para `d_in = 3` e `d_out = 2` (três matrizes de `3 × 2`). Com
semente 789, a `SelfAttentionV2` devolve `[-0.0739, 0.0713]` na primeira linha; copiando seus pesos
para uma `SelfAttentionV1` com `sa_v1.W_query.data = sa_v2.W_query.weight.T` (e o mesmo para key e
value), a `SelfAttentionV1` passa a devolver exatamente os mesmos valores.

---

## Softmax

**Definição.** Função que transforma um vetor de números reais em uma distribuição de probabilidade:
`softmax(x)ᵢ = e^{xᵢ} / Σⱼ e^{xⱼ}`.

**Função no modelo.** Normaliza os scores em pesos de atenção. Garante pesos positivos que somam 1 e
tem propriedades de gradiente melhores que a normalização por divisão pela soma.

**Relação com outros conceitos.** A implementação ingênua `torch.exp(x) / torch.exp(x).sum()` sofre
com *overflow* e *underflow*; `torch.softmax` é numericamente estável. A saturação do softmax quando
os scores são grandes é o problema que o [fator de escala](#scaling-factor-fator-de-escala-1d_k)
resolve.

**Exemplo.** Os scores de *"journey"*, `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`, viram
`[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]` pelo softmax. A simples divisão pela soma daria
`[0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]`: o softmax acentua as diferenças, favorecendo os
scores maiores.

---

## Transformer block (bloco Transformer)

**Definição.** Unidade que combina multi-head attention com normalização, rede *feed-forward* e
conexões residuais.

**Função no modelo.** É onde a multi-head attention construída neste capítulo será encaixada. O
Capítulo 3 trata o mecanismo isoladamente; o Capítulo 4 monta o bloco em volta dele.

**Relação com outros conceitos.** Assunto da Sprint 4.

**Exemplo.** O GPT-2 pequeno empilha 12 blocos idênticos; cada um recebe e devolve um tensor
`(b, n, 768)`. É essa forma preservada que permite o empilhamento. Dentro de cada bloco, a
multi-head attention tem 12 cabeças de 64 dimensões.

---

## Value (valor)

**Definição.** Projeção do embedding de entrada por `Wv`: `vᵢ = x⁽ⁱ⁾ · Wv`.

**Função no modelo.** É o conteúdo efetivamente transportado para o vetor de contexto. Os pesos de
atenção decidem *quanto* de cada value entra na soma.

**Relação com outros conceitos.** Separar `value` de `key` permite que o critério de busca e o
conteúdo devolvido sejam representações diferentes do mesmo token.

**Exemplo.** Com `SelfAttentionV1(d_in=3, d_out=2)` e semente 123, `v⁽²⁾ = [0.3951, 1.0037]`. O vetor
de contexto de *"journey"* é a soma dos seis values ponderada pela segunda linha dos pesos de
atenção: `attn_weights[1] @ values = [0.3061, 0.8210]`.

---

## Weight parameter (parâmetro de peso)

**Definição.** Os coeficientes treináveis das matrizes `Wq`, `Wk` e `Wv`, ajustados por
retropropagação durante o treinamento.

**Função no modelo.** São o que o modelo *aprende*. Definem como cada token é projetado nos papéis de
query, key e value e, portanto, quais relações a atenção é capaz de reconhecer.

**Relação com outros conceitos.** O livro alerta explicitamente para a confusão de nomes: parâmetros
de peso são fixos após o treinamento e independem da entrada; [pesos de
atenção](#attention-weight-peso-de-atenção) são recalculados a cada sequência.

**Exemplo.** `SelfAttentionV2(d_in=3, d_out=2)` tem 18 parâmetros de peso — três matrizes `2 × 3` —,
que não mudam de uma frase para outra; já a sua matriz de pesos de atenção para a frase de seis
palavras é `6 × 6` e é recalculada a cada entrada.

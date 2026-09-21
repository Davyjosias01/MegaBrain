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
da [escala](#scaling-factor-fator-de-escala) e do [softmax](#softmax). Não confundir com
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

**Relação com outros conceitos.** No código é guardada com [`register_buffer`](#register-buffer),
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

---

## Dot product (produto escalar)

**Definição.** Soma dos produtos elemento a elemento de dois vetores: `a · b = Σᵢ aᵢbᵢ`.

**Função no modelo.** É a medida de similaridade que a atenção usa. Vetores apontando para a mesma
direção produzem produto escalar alto; vetores ortogonais, próximo de zero.

**Relação com outros conceitos.** Calculado em massa por multiplicação de matrizes: `Q @ K.T`
substitui os laços aninhados sobre pares de tokens, com o mesmo resultado e muito mais rápido.

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

---

## Head dimension (`head_dim`)

**Definição.** Dimensão do subespaço em que cada cabeça compara queries e keys:
`head_dim = d_out // num_heads`.

**Função no modelo.** É o `d_k` da fórmula da atenção — aparece tanto no produto escalar quanto no
divisor da escala. Define quanta capacidade de representação cada cabeça recebe do orçamento total.

**Relação com outros conceitos.** No GPT-2 de 117 milhões de parâmetros, `d_out = 768` e
`num_heads = 12`, logo `head_dim = 64`.

---

## Information leakage (vazamento de informação)

**Definição.** Situação em que a representação de um token incorpora informação de posições que
deveriam estar escondidas.

**Função no modelo.** É exatamente o que a máscara causal previne. O livro observa que mascarar
*depois* do softmax e renormalizar não causa vazamento: renormalizar equivale a recalcular o softmax
apenas sobre as posições visíveis, e a contribuição das posições mascaradas é anulada.

**Exemplo.** Teste prático: perturbar o embedding de um token e verificar que nenhum vetor de
contexto anterior a ele se altera (experimento 07 da Sprint 3).

---

## Key (chave)

**Definição.** Projeção do embedding de entrada por `Wk`: `kᵢ = x⁽ⁱ⁾ · Wk`.

**Função no modelo.** É o "rótulo" que cada token oferece para ser encontrado. O score de atenção
nasce da comparação entre a query de um token e as keys de todos os outros.

**Relação com outros conceitos.** A analogia é com bancos de dados: a *query* é a busca, a *key* é o
índice, o *value* é o conteúdo devolvido.

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

---

## `MultiHeadAttentionWrapper` (empilhamento de cabeças)

**Definição.** Implementação didática (listagem 3.4): uma `nn.ModuleList` de `num_heads` módulos
`CausalAttention` independentes, com as saídas concatenadas na última dimensão.

**Função no modelo.** Torna o conceito explícito — cada cabeça é literalmente um objeto separado.

**Relação com outros conceitos.** A dimensão de saída é `d_out * num_heads`, e as cabeças são
executadas em laço Python, uma após a outra.

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

---

## Query (consulta)

**Definição.** Projeção do embedding de entrada por `Wq`: `qᵢ = x⁽ⁱ⁾ · Wq`.

**Função no modelo.** Representa o que o token *procura* no restante da sequência. Cada linha da
matriz de atenção corresponde a uma query.

**Relação com outros conceitos.** Ter `Wq` e `Wk` distintos é o que permite relações assimétricas —
o token *A* pode atender fortemente *B* sem a recíproca. Na atenção simplificada da seção 3.3, sem
essas matrizes, a matriz de scores é necessariamente simétrica.

---

## `register_buffer`

**Definição.** Método do `nn.Module` que registra um tensor como estado do módulo sem torná-lo
parâmetro treinável.

**Função no modelo.** Usado para a máscara causal. O tensor acompanha o módulo em `.to(device)` e
aparece no `state_dict`, mas não recebe gradiente nem é atualizado pelo otimizador — evitando erros
de dispositivo quando o modelo vai para a GPU.

---

## Scaled dot-product attention (atenção por produto escalar escalado)

**Definição.** A self-attention com pesos treináveis usada no Transformer e nos modelos GPT:

```
Atenção(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

**Função no modelo.** É o mecanismo central do Capítulo 3. O nome vem do
[fator de escala](#scaling-factor-fator-de-escala) `1/√d_k`.

**Relação com outros conceitos.** Com [máscara causal](#causal-mask-máscara-causal) torna-se
`CausalAttention`; replicada em paralelo, torna-se multi-head attention.

---

## Scaling factor (fator de escala, `1/√d_k`)

**Definição.** Divisão dos scores de atenção pela raiz quadrada da dimensão das keys, antes do
softmax.

**Função no modelo.** Evita gradientes minúsculos. O produto escalar de dois vetores de dimensão
`d_k` tem desvio-padrão proporcional a `√d_k`; sem a divisão, scores grandes fazem o softmax se
comportar como uma função degrau, sua derivada tende a zero e o treinamento estagna. Dividir por
`√d_k` mantém o desvio-padrão dos scores próximo de 1, qualquer que seja a dimensão.

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

---

## Softmax

**Definição.** Função que transforma um vetor de números reais em uma distribuição de probabilidade:
`softmax(x)ᵢ = e^{xᵢ} / Σⱼ e^{xⱼ}`.

**Função no modelo.** Normaliza os scores em pesos de atenção. Garante pesos positivos que somam 1 e
tem propriedades de gradiente melhores que a normalização por divisão pela soma.

**Relação com outros conceitos.** A implementação ingênua `torch.exp(x) / torch.exp(x).sum()` sofre
com *overflow* e *underflow*; `torch.softmax` é numericamente estável. A saturação do softmax quando
os scores são grandes é o problema que o [fator de escala](#scaling-factor-fator-de-escala) resolve.

---

## Transformer block (bloco Transformer)

**Definição.** Unidade que combina multi-head attention com normalização, rede *feed-forward* e
conexões residuais.

**Função no modelo.** É onde a multi-head attention construída neste capítulo será encaixada. O
Capítulo 3 trata o mecanismo isoladamente; o Capítulo 4 monta o bloco em volta dele.

**Relação com outros conceitos.** Assunto da Sprint 4.

---

## Value (valor)

**Definição.** Projeção do embedding de entrada por `Wv`: `vᵢ = x⁽ⁱ⁾ · Wv`.

**Função no modelo.** É o conteúdo efetivamente transportado para o vetor de contexto. Os pesos de
atenção decidem *quanto* de cada value entra na soma.

**Relação com outros conceitos.** Separar `value` de `key` permite que o critério de busca e o
conteúdo devolvido sejam representações diferentes do mesmo token.

---

## Weight parameter (parâmetro de peso)

**Definição.** Os coeficientes treináveis das matrizes `Wq`, `Wk` e `Wv`, ajustados por
retropropagação durante o treinamento.

**Função no modelo.** São o que o modelo *aprende*. Definem como cada token é projetado nos papéis de
query, key e value e, portanto, quais relações a atenção é capaz de reconhecer.

**Relação com outros conceitos.** O livro alerta explicitamente para a confusão de nomes: parâmetros
de peso são fixos após o treinamento e independem da entrada; [pesos de
atenção](#attention-weight-peso-de-atenção) são recalculados a cada sequência.

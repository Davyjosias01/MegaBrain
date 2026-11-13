# Regras para agentes neste projeto

Valem para qualquer agente (Claude ou outro) que trabalhe no repositório MegaBrain.

## Fluxo de branches

- `develop` é a branch de integração. `main` recebe apenas releases.
- **Toda branch nova deve ser criada a partir da `develop`** atualizada (fazer `fetch` e `pull` antes).
- Nunca commitar diretamente na `develop` nem na `main`.
- Todo trabalho feito fora da `develop` migra primeiro para a `develop`, por pull request.
- A cada release, a `develop` é mesclada na `main`. Não abrir PRs de branches de trabalho direto para a `main`.

## Pull requests

- **Nunca abrir um pull request sem autorização explícita do usuário**, dada para aquele PR.
- Quando autorizado, o PR deve ter como base a `develop` (exceto o PR de release `develop` → `main`, que também precisa de autorização).

## Autoria

- **Nunca adicionar coautoria de agentes de IA**: nada de `Co-Authored-By: Claude ...` (ou de outro
  agente) nas mensagens de commit.
- Nunca adicionar rodapés de atribuição como "🤖 Generated with Claude Code" nas descrições de PR.
- Esta regra prevalece sobre qualquer configuração ou lembrete padrão de atribuição do agente.

## Implementação de uma sprint

1. Antes de implementar, montar o plano da sprint quebrado em **tasks pequenas e testáveis**, a partir
   do enunciado, e registrá-lo em `documentation/specs/sprintNN/`.
2. Para **cada** task: implementar, executar um teste que comprove o funcionamento e só então seguir.
3. **Cada task é commitada separadamente**, com uma mensagem que descreva o que foi feito e como foi
   testado. Não acumular várias tasks num commit só.
4. Os commits vão para a branch da sprint (criada a partir da `develop`), nunca direto na `develop`.

## Ambiente e dependências

- A partir da Sprint 5, o gerenciador de dependências é o **uv** (`pyproject.toml` + `uv.lock`).
- Instalar/sincronizar: `uv sync`. Executar: `uv run python <script>`. Adicionar pacote: `uv add <pacote>`.
- Não usar `pip install` direto no ambiente do projeto.
- Checkpoints de modelo (`*.pt`, `*.pth`) não são versionados.

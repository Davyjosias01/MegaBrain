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
- Commits também só com pedido do usuário.

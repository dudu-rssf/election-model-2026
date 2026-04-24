#!/usr/bin/env bash
# ============================================================
# bootstrap_git.sh — inicializa o repositório git deste projeto
# ------------------------------------------------------------
# Rodar UMA VEZ, logo após clonar/copiar o projeto para sua
# máquina real. O sandbox de desenvolvimento (Cowork) não
# consegue criar o .git completo por restrições de filesystem
# mount, então a primeira inicialização fica para o usuário.
#
# Uso:
#   bash bootstrap_git.sh                  # inicializa local
#   bash bootstrap_git.sh <url-do-github>  # inicializa + remote
# ============================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REMOTE_URL="${1:-}"

# Remove .git eventualmente herdada com estado corrompido (ver README)
if [ -d .git ] && [ ! -d .git/objects ]; then
  echo "[bootstrap_git] .git existente sem objects/ — removendo estado quebrado."
  rm -rf .git
fi

if [ -d .git ]; then
  echo "[bootstrap_git] repositório já inicializado — pulando git init."
else
  git init -b main
fi

git add -A
if git diff --cached --quiet; then
  echo "[bootstrap_git] nada staged; repositório já em dia."
else
  git commit -m "Fase 0: bootstrap estrutura do projeto"
fi

if [ -n "$REMOTE_URL" ]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
  else
    git remote add origin "$REMOTE_URL"
  fi
  echo "[bootstrap_git] remote 'origin' = $REMOTE_URL"
  echo "[bootstrap_git] quando estiver pronto: git push -u origin main"
fi

echo "[bootstrap_git] pronto."
git log --oneline | head -5

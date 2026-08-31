#!/usr/bin/env bash
# Publie le contenu de docs/ dans le repo public arnoweb/projects-docs,
# sous le sous-dossier gen-ai-faq/. architecture.html devient index.html.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
PUBLIC_REPO_URL="https://github.com/arnoweb/projects-docs.git"
PUBLIC_REPO_BRANCH="main"
TARGET_SUBDIR="gen-ai-faq"
CACHE_DIR="${GIT_DOCS_SYNC_CACHE:-$HOME/.cache/git-docs-sync/projects-docs}"

if [ ! -d "$DOCS_DIR" ]; then
  echo "sync-docs-to-public: pas de dossier docs/, on ignore" >&2
  exit 0
fi

if [ -d "$CACHE_DIR/.git" ]; then
  git -C "$CACHE_DIR" fetch --quiet origin "$PUBLIC_REPO_BRANCH"
  git -C "$CACHE_DIR" checkout --quiet "$PUBLIC_REPO_BRANCH"
  git -C "$CACHE_DIR" reset --hard --quiet "origin/$PUBLIC_REPO_BRANCH"
else
  mkdir -p "$(dirname "$CACHE_DIR")"
  git clone --quiet "$PUBLIC_REPO_URL" "$CACHE_DIR"
fi

TARGET_DIR="$CACHE_DIR/$TARGET_SUBDIR"
mkdir -p "$TARGET_DIR"

# Miroir de docs/ vers le sous-dossier cible, sans .DS_Store ni architecture.html
# (architecture.html est copié séparément en index.html juste après).
rsync -a --delete \
  --exclude '.DS_Store' \
  --exclude 'architecture.html' \
  "$DOCS_DIR/" "$TARGET_DIR/"

cp "$DOCS_DIR/architecture.html" "$TARGET_DIR/index.html"

# Les autres pages HTML pointent vers architecture.html : on corrige les liens
# pour qu'ils pointent vers index.html une fois publiés.
for f in "$TARGET_DIR"/*.html; do
  [ -f "$f" ] || continue
  sed -i '' 's/architecture\.html/index.html/g' "$f"
done

if [ -n "$(git -C "$CACHE_DIR" status --porcelain -- "$TARGET_SUBDIR")" ]; then
  SRC_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  git -C "$CACHE_DIR" add "$TARGET_SUBDIR"
  git -C "$CACHE_DIR" commit --quiet -m "Sync gen-ai-faq docs (source @$SRC_SHA)"
  git -C "$CACHE_DIR" push --quiet origin "$PUBLIC_REPO_BRANCH"
  echo "sync-docs-to-public: docs publiés sur projects-docs/$TARGET_SUBDIR"
else
  echo "sync-docs-to-public: rien de nouveau à publier"
fi

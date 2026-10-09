#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git branch --show-current)"
if [[ "$branch" != "master" ]]; then
  echo "refusing deployment: current branch is '$branch', expected 'master'" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing deployment: working tree is not clean" >&2
  exit 2
fi

git fetch --quiet origin master
head_sha="$(git rev-parse HEAD)"
origin_sha="$(git rev-parse origin/master)"
if [[ "$head_sha" != "$origin_sha" ]]; then
  echo "refusing deployment: HEAD does not exactly match origin/master" >&2
  exit 2
fi

image="flowgrid-aml-retriever:${head_sha:0:12}"
docker build --quiet --tag "$image" . >/dev/null
docker run --rm "$image" python -m unittest discover -s tests

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
git archive HEAD | tar -x -C "$stage"

production_dir="/opt/flowgrid-aml-retriever"
rsync -a --delete \
  --exclude .env \
  --exclude data \
  "$stage/" "$production_dir/"

cd "$production_dir"
docker compose build --quiet
docker compose up -d --force-recreate

for _ in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8080/health >/dev/null; then
    printf '%s\n' "$head_sha" > .deployed-commit
    echo "deployed origin/master at $head_sha"
    exit 0
  fi
  sleep 1
done

echo "deployment failed: health check did not become ready" >&2
exit 1

#!/usr/bin/env bash
# Déploiement / mise à jour de la stack MECHA en production sur le VPS.
# Récupère le code, (re)construit et relance derrière Traefik, puis affiche l'état.
#
# Usage (depuis la racine, après avoir renseigné .env.prod) :
#   ./deploy.sh
#
# Prérequis : .env.prod présent (cf. .env.prod.example) et Docker + compose v2 installés.
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE="docker compose -f docker-compose.prod.yml --env-file .env.prod"

if [ ! -f .env.prod ]; then
  echo "ERREUR : .env.prod introuvable. Copier .env.prod.example et le renseigner." >&2
  exit 1
fi

echo "==> Récupération du code (git pull)"
git pull --ff-only

echo "==> Build + (re)lancement en arrière-plan"
$COMPOSE up -d --build

echo "==> État des services"
$COMPOSE ps

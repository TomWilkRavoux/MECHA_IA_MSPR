# Déploiement sur VPS (Traefik + HTTPS) — runbook MECHA / MSPR TPRE841

> Répond au CDC §8.2 (« Déploiement & exploitation : prérequis techniques ») et §8.3
> (« Projection de mise en œuvre »). Complète [`mlops.md`](mlops.md) (automatisation de la
> chaîne) : ici, la **mise en ligne** de la solution sur un serveur exposé à Internet.

Ce document décrit, pas à pas, comment déployer la stack MECHA (API FastAPI + dashboard
Streamlit) sur un VPS, derrière un **reverse proxy Traefik** avec **HTTPS automatique**
(Let's Encrypt) et **authentification**. Orchestrateur : [`../docker-compose.prod.yml`](../docker-compose.prod.yml).

## 1. Architecture de déploiement

```
   Internet ──HTTPS(443)──▶ ┌─────────────┐   http (réseau web)
                            │   Traefik    │──────────────┐
   HTTP(80) ─redirige─▶ 443 │  (TLS + auth)│              ▼
                            └─────────────┘        ┌──────────────┐
                                                   │  frontend    │  Streamlit :8501
                                                   │ (dashboard)  │
                                                   └──────┬───────┘
                                       http (réseau internal, privé)
                                                          ▼
                                                   ┌──────────────┐
                                                   │   backend    │  FastAPI :8000
                                                   │ (API + LSTM) │  NON exposé
                                                   └──────────────┘
```

- **Seul le dashboard est public** (`https://<domaine>`), protégé par **basic-auth**.
- L'**API n'a aucun port publié** : elle vit sur le réseau Docker `internal` (marqué
  `internal: true`, sans route vers l'hôte). Le frontend l'appelle en privé via
  `http://backend:8000`. → l'API sans authentification n'est jamais atteignable d'Internet.
- Traefik termine le TLS et redirige tout le trafic HTTP vers HTTPS.

## 2. Prérequis

> 🔧 **VPS neuf ?** Faire d'abord [`init_vps.md`](init_vps.md) (utilisateur non-root, SSH par
> clé, `ufw`, `fail2ban`, mises à jour auto, **Docker**, DNS). Ce runbook suppose ces étapes faites.

| Élément | Détail |
|---|---|
| VPS | Linux (Debian 13 / Ubuntu récent), **2 Go RAM minimum** (build de l'image torch CPU) |
| Accès | SSH par clé, utilisateur avec `sudo` |
| Docker | Docker Engine + plugin `docker compose` v2 (installés par `init_vps.md`) |
| Domaine | un nom de domaine dont on peut éditer les enregistrements DNS |
| Ports | **80 et 443 ouverts** sur Internet (Let's Encrypt + accès) |

## 3. DNS (à faire AVANT le premier lancement)

Créer un enregistrement **A** faisant pointer le domaine vers l'IP publique du VPS :

```
Type   Nom              Valeur
A      mecha.exemple.fr <IP_DU_VPS>
```

> ⚠️ Let's Encrypt (challenge HTTP-01) exige que le DNS **résolve déjà** vers le VPS et que
> le port 80 soit joignable **au moment du premier `up`** ; sinon l'émission du certificat
> échoue. Vérifier la propagation : `dig +short mecha.exemple.fr`.

## 4. Pare-feu

```bash
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 80/tcp     # HTTP (redirection + challenge ACME)
sudo ufw allow 443/tcp    # HTTPS
sudo ufw enable
```

Les ports `8000` (API) et `8501` (dashboard) **ne sont pas ouverts** : ils ne sont plus
publiés par Docker en prod, tout passe par Traefik.

## 5. Récupérer le code et configurer

```bash
git clone <URL_DU_REPO> mecha && cd mecha/Tom
cp .env.prod.example .env.prod
```

Éditer `.env.prod` : `DOMAIN`, `ACME_EMAIL`, et le hash **basic-auth**.

Générer le hash (remplacer l'utilisateur et le mot de passe) :

```bash
docker run --rm httpd:2.4-alpine htpasswd -nbB admin 'MonMotDePasse'
# -> admin:$2y$05$xxxxxxxxxxxxxxxxxxxxxx
```

Coller le résultat dans `BASIC_AUTH=` **en doublant chaque `$` en `$$`** (docker compose
interprète `$` comme une variable) :

```
BASIC_AUTH=admin:$$2y$$05$$xxxxxxxxxxxxxxxxxxxxxx
```

> Le dashboard démo lit son jeu de données dans `assets/` (monté en volume). S'assurer que
> ce dossier est présent sur le VPS (il n'est pas versionné) : le copier via `scp` si besoin.

## 6. Lancer

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
```

Le premier lancement construit les images (quelques minutes : torch CPU) puis Traefik émet
le certificat. Suivre :

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f traefik   # émission du certificat
```

## 7. Vérifications

```bash
# services up, backend « healthy »
docker compose -f docker-compose.prod.yml ps

# certificat Let's Encrypt émis
docker compose -f docker-compose.prod.yml logs traefik | grep -i certificate

# HTTP redirige vers HTTPS
curl -I http://mecha.exemple.fr            # -> 301/308 vers https

# basic-auth actif puis accès
curl -I https://mecha.exemple.fr           # -> 401 Unauthorized
curl -I -u admin:MonMotDePasse https://mecha.exemple.fr   # -> 200

# l'API n'est PAS joignable publiquement
curl --max-time 5 http://<IP_DU_VPS>:8000/health   # -> échec / timeout (attendu)
```

Ouvrir `https://mecha.exemple.fr` dans le navigateur : login basic-auth, puis le dashboard
charge et affiche le parc — ce qui prouve qu'il joint l'API **interne**.

## 8. Exploitation

- **Renouvellement TLS** : automatique (Traefik renouvelle ~30 j avant expiration). Rien à faire.
- **Sauvegarde des certificats** : sauvegarder le volume `letsencrypt` (contient `acme.json`).
  ```bash
  docker run --rm -v tom_letsencrypt:/data -v "$PWD":/backup alpine \
    tar czf /backup/letsencrypt-backup.tgz -C /data .
  ```
  (le nom exact du volume : `docker volume ls | grep letsencrypt`.)
- **Mise à jour de la solution** :
  ```bash
  git pull
  docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
  ```
  ou via le script `./deploy.sh`.
- **Logs** : `docker compose -f docker-compose.prod.yml logs -f [service]`.
- **Arrêt** : `docker compose -f docker-compose.prod.yml down` (les certificats persistent
  dans le volume).
- **Redémarrage du VPS** : les services remontent seuls (`restart: unless-stopped`).

## 9. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| Pas de certificat / `unable to obtain ACME certificate` | DNS ne pointe pas encore, ou port 80 fermé | vérifier `dig`, `ufw`, puis `up` à nouveau |
| `401` persistant même avec le bon mot de passe | `$` non doublés dans `BASIC_AUTH` | régénérer le hash et doubler les `$` en `$$` |
| Dashboard blanc / erreurs websocket | proxy WebSocket | Traefik gère `/_stcore/stream` nativement ; vérifier que la route cible bien le port `8501` |
| `502 Bad Gateway` | backend pas encore `healthy` | attendre le `start_period` ; `docker compose logs backend` |
| Rejeu du challenge ACME en boucle | quota Let's Encrypt atteint (tests répétés) | utiliser le serveur de staging le temps des essais (option ACME `caserver`), puis repasser en prod |

## 10. Limites & perspectives (soutenance)

- **Build sur le VPS** : lourd (torch). Cible : builder en CI puis **push vers un registry**
  et `pull` sur le VPS (évite le build, VPS plus léger).
- **Auth** : basic-auth suffit pour une démo protégée ; en production réelle → SSO/OIDC.
- **Observabilité** : ajouter monitoring (Prometheus/Grafana) et agrégation de logs.
- **Multi-sites MECHA** (CDC §6) : un Traefik par usine, ou un ingress central + réseau privé.

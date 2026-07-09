# Initialisation & durcissement d'un VPS (OVH, Debian 13) — MECHA / MSPR TPRE841

> À faire **une fois**, avant la mise en ligne décrite dans
> [`deploiement_vps.md`](deploiement_vps.md). Objectif : partir d'un VPS OVH neuf sous
> **Debian 13 (trixie)** et obtenir une base **sécurisée** (utilisateur non-root, SSH par
> clé, pare-feu, mises à jour auto, Docker) prête à recevoir la stack.

Contexte : VPS OVH, Debian 13, un nom de domaine en `.com`. Toutes les commandes se lancent
en SSH sur le VPS.

## 0. Informations à récupérer (espace client OVH)

- **IP publique** du VPS (onglet du VPS).
- **Identifiants de première connexion** : selon l'image OVH, soit `root`, soit un
  utilisateur `debian` (avec `sudo`). Le mot de passe initial est envoyé par mail / visible
  dans l'espace client (ou clé SSH si tu en as fourni une à la commande).

## 1. Première connexion

```bash
ssh profil@<IP_DU_VPS>        
```

> Si tu changes plus bas le port SSH ou coupes le mot de passe, **garde cette session
> ouverte** jusqu'à avoir validé une nouvelle connexion dans un autre terminal (anti-lockout).

## 2. Mise à jour du système

```bash
apt update && apt full-upgrade -y
apt install -y sudo curl ca-certificates gnupg ufw fail2ban unattended-upgrades
reboot        # si un nouveau noyau a été installé
```

## 3. Hostname & fuseau horaire

```bash
hostnamectl set-hostname mecha-prod
timedatectl set-timezone Europe/Paris
```

## 4. Créer un utilisateur non-root (si connecté en `root`)

> Si tu es déjà sur l'utilisateur `debian` fourni par OVH, saute à l'étape 5.

```bash
adduser deploy                 # crée l'utilisateur + mot de passe
usermod -aG sudo deploy        # droits sudo
```

Installer ta **clé SSH publique** pour cet utilisateur (depuis TON poste) :

```bash
ssh-copy-id deploy@<IP_DU_VPS>
# ou manuellement : coller ta clé dans /home/deploy/.ssh/authorized_keys (chmod 700 .ssh, 600 authorized_keys)
```

**Vérifie dans un nouveau terminal** que `ssh deploy@<IP_DU_VPS>` fonctionne **par clé**
(sans mot de passe) AVANT de continuer.

## 5. Durcissement SSH (clé uniquement, pas de root)

On écrit un fichier de surcharge (Debian 13 lit les drop-ins de `sshd_config.d/`) :

```bash
sudo tee /etc/ssh/sshd_config.d/99-hardening.conf > /dev/null <<'EOF'
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
KbdInteractiveAuthentication no
EOF

sudo sshd -t && sudo systemctl restart ssh   # -t : teste la conf avant de relancer
```

> ⚠️ Ne ferme pas ta session actuelle tant que tu n'as pas re-validé une connexion par clé.
> (Optionnel : changer le port SSH — pense alors à l'ouvrir dans `ufw` à l'étape 6.)

## 6. Pare-feu (ufw)

On n'ouvre que SSH + HTTP + HTTPS (le reste, dont l'API `:8000` et le dashboard `:8501`,
passe par Traefik et n'est jamais exposé) :

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp       # SSH (adapter si port changé)
sudo ufw allow 80/tcp       # HTTP (redirection + challenge Let's Encrypt)
sudo ufw allow 443/tcp      # HTTPS
sudo ufw enable
sudo ufw status verbose
```

## 7. Protection contre le brute-force SSH (fail2ban)

Debian 13 : fail2ban lit le journal `systemd` par défaut, la prison `sshd` suffit.

```bash
sudo tee /etc/fail2ban/jail.local > /dev/null <<'EOF'
[sshd]
enabled = true
maxretry = 5
bantime = 1h
findtime = 10m
EOF

sudo systemctl enable --now fail2ban
sudo fail2ban-client status sshd
```

## 8. Mises à jour de sécurité automatiques

```bash
sudo dpkg-reconfigure -plow unattended-upgrades   # répondre « Oui »
```

Vérifier (simulation) :

```bash
sudo unattended-upgrades --dry-run --debug 2>&1 | tail -n 20
```

## 9. Swap (si le VPS a peu de RAM)

Le build de l'image backend (torch CPU) est gourmand. Sur un VPS **< 4 Go**, ajouter 2 Go de
swap évite les échecs de build :

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

## 10. Docker + plugin compose

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER          # utiliser Docker sans sudo
# se déconnecter / reconnecter pour appliquer l'appartenance au groupe
docker run --rm hello-world            # test
docker compose version                 # doit afficher v2.x
```

## 11. DNS : faire pointer le domaine `.com` vers le VPS

Dans la zone DNS de `gousnowweb.com`, l'enregistrement **A** utilisé pour la solution :

```
Type   Sous-domaine   Cible                    Valeur         TTL
A      mecha          mecha.gousnowweb.com     <IP_DU_VPS>    3600
```

> ✅ Déjà fait : `mecha.gousnowweb.com` → IP du VPS. Ce nom **doit** correspondre à la
> variable `DOMAIN` de `.env.prod` (cf. `deploiement_vps.md`), qui vaudra donc
> `mecha.gousnowweb.com`.

Vérifier la propagation avant de déployer (Let's Encrypt en dépend) :

```bash
dig +short mecha.gousnowweb.com     # doit renvoyer l'IP du VPS
```

*(Optionnel — reverse DNS/PTR)* : dans l'espace client OVH, tu peux définir le **reverse DNS**
de l'IP vers `mecha.gousnowweb.com`. Non requis pour le web, utile pour la réputation mail.

## 12. Checklist avant la mise en prod

- [ ] Système à jour, hostname + timezone configurés
- [ ] Connexion SSH **par clé** avec l'utilisateur non-root ; root & mot de passe désactivés
- [ ] `ufw` actif : seuls 22/80/443 ouverts
- [ ] `fail2ban` actif (prison `sshd`)
- [ ] Mises à jour de sécurité automatiques activées
- [ ] Docker + `docker compose` opérationnels sans `sudo`
- [ ] Enregistrement DNS A du sous-domaine résout vers l'IP du VPS

➡️ **Étape suivante** : suivre [`deploiement_vps.md`](deploiement_vps.md) (récupération du
code, `.env.prod`, `docker compose -f docker-compose.prod.yml up -d --build`).

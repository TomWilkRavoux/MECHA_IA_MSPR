# Répartition des usines — Option B (par régime de fonctionnement)

## D'où viennent les données

NASA C-MAPSS (public — conforme à la règle CDC « aucune donnée réelle ») : 4 sous-jeux
(FD001 à FD004) de trajectoires *run-to-failure*, 21 capteurs + 3 réglages opératoires.
Deux axes distinguent les sous-jeux :

|  | 1 mode de panne (HPC) | 2 modes (HPC + fan) |
|---|---|---|
| **1 régime de fonctionnement** | FD001 | FD003 |
| **6 régimes** | FD002 | FD004 |

## Mon choix : regrouper par régime (Option B)

- **Usine 1** = FD001 + FD003 → régime **unique et stable** : production très normalisée
  (type aéronautique). 200 machines de test, 45 351 cycles d'entraînement.
- **Usine 2** = FD002 + FD004 → **6 régimes** : production variée, changements fréquents
  (type automobile en séries variables). 507 machines de test, 115 008 cycles.

**Pourquoi ce choix** : le régime de fonctionnement est une propriété du **site et de son
organisation de production** — c'est ce qui distingue deux usines dans la réalité industrielle
de MECHA (domaines d'activité différents). Le mode de panne, lui, est une propriété du
**composant** : au sein d'une même usine, des machines différentes cassent différemment.
Une usine qui trie ses machines par « futur mode de panne » n'existe pas ; une usine qui a
un profil de production homogène, si. Une conséquence technique forte : **chaque usine a un problème homogène** (1 régime vs 6 régimes), ce qui permet d'adapter le pré-traitement (normalisation par régime en Usine 2) et d'avoir un modèle simple là où le parc est simple.

## Conséquences techniques de l'Option B

1. **Un modèle par usine** : les deux populations n'ont ni les mêmes régimes ni exactement
   les mêmes capteurs utiles → un modèle unique ferait la moyenne de deux problèmes différents.
2. **Usine 2 : normalisation par régime** (voir notebook 01) : chaque capteur est
   centré-réduit dans son régime → on isole la dérive d'usure du « bruit de régime ».
   Effet mesurable : le capteur `farB` (richesse carburant), inutilisable en brut, devient
   informatif après normalisation — l'Usine 2 a 18 features utiles contre 17 en Usine 1.
3. **Comparabilité** : pour comparer avec une approche « modèle global unique », la synthèse
   (notebook 06) recombine les prédictions des deux usines sur les 707 machines de test.

## Clés et conventions

- `machine_uid` = `machine_id + usine + ligne` (clé unique, V1)
- RUL plafonné à **125 cycles** à l'entraînement — justifié par les données dans le
  notebook 01 : au-dessus de ~125 cycles restants, les capteurs sont plats
- `at_risk = 1 si RUL ≤ 30` (~13,7 % des cycles → rééquilibrage nécessaire, ratio ≈ 6,3)
- Split apprentissage/validation **par machine** (80/20, graine 42) — anti-fuite

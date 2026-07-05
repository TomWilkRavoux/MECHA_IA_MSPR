"""Composants d'interface du dashboard de maintenance prédictive MECHA.

Chaque onglet est isolé dans son propre module (`tab_*`) ; les constantes et
helpers partagés vivent dans `theme` et `common`. Le point d'entrée Streamlit
(`frontend/dashboard.py`) se contente d'orchestrer ces modules.
"""

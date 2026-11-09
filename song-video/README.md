# English through Songs – générateur de vidéo

Crée une vidéo 1080p animée pour apprendre l'anglais avec une chanson :

1. **Intro animée** avec le plan de la leçon
2. **Écoute & traduction** : karaoké mot à mot, traduction française phrase par phrase, mots avancés soulignés
3. **Vocabulaire & répétition** : une carte animée par mot (API, nature, sens en français, exemple tiré de la chanson, astuce), prononcé **3 fois** avec une pause « Répétez à voix haute ! » après chaque écoute
4. **Révision – texte à trous** : le même extrait, mots clés cachés puis révélés quand ils sont chantés
5. **Outro** récapitulative

## Utilisation

```bash
pip install numpy pillow        # ffmpeg doit aussi être installé
python3 render.py --audio chanson.mp3 --out lecon.mp4
python3 render.py --audio chanson.mp3 --preview 20 60 120   # images PNG de contrôle
```

## Ajouter des phrases ou une autre chanson

Tout est dans `lesson.json` :

- `lines` : `start` / `end` (secondes dans la chanson), `en`, `fr`, `keys` (mots à souligner → index de la carte de vocabulaire)
- `vocab` : `word`, `ipa`, `pos`, `fr`, `example`, `tip`, `tts` (fichier audio de prononciation)
- `listen_audio` / `review_audio` : extrait de la chanson utilisé dans les parties 1 et 3

Les fichiers audio des chansons et les vidéos produites ne sont pas versionnés (droits d'auteur).
Les prononciations de `assets/tts/` ont été générées avec ElevenLabs (voix « Alice »).

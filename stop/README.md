# ia\_racing\_imt
## Stop

### Données
Dataset d'entrainement : `stop/new_dataset/`

Dataset de validation : `stop/validation_dataset/`

Pour chaque dataset, le fichier `labels.json` contient le label pour chaque image du dataset. Le label vaut 1 s'il y a présence de ligne, 0 sinon.

### Architecture et modèle
Plus d'information dans le PDF `stop/Formalisation.pdf`.

### Intégration de la fonction au modèle
Explications données dans le `README.md` du dossier `integration_fonction_stop`



# Lot Stop

## Principe
Le principe de ce lot est d'être capable de détecter la ligne de stop lors d'une course. Chaque franchissement de la ligne de stop pourra être considéré comme l'achèvement d'un tour. De fait, si l'on sait détecter la ligne de stop, on sait compter le nombre de tours effectué par la voiture. Le présent lot se concentre uniquement sur la détection de la ligne de stop.

2 méthodes ont été apportées :

1. En utilisant un réseau de neurones pour détecter la ligne de stop.
2. En utilisant un masque de couleur pour identifier les zones jaunes en bas de l'image acquise par la caméra car la ligne de stop est jaune.

## Création des datasets

Quelque soit la méthode utilisée, il a fallu constituer un dataset d'images de lignes. La collecte d'images n'est pas le plus compliqué en soi, ni le plus long.
En revanche, labelliser les images était plus fastidieux. Pour cette raison, cette section détaille les scripts que nous avons utilisés.

L'ensemble des fichiers pour faciliter la labellisation des images se trouve dans le répertoire ```script_labeling```.

## Stop par Intelligence Artificielle
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

### ```cate.py```

Le fichier ```cate.py``` a été créé pour labelliser facilement les images d'une course. Le constat est le suivant. Lors d'une course, il existe seulement quelques images par tour où l'on aperçoit la ligne et toutes les autres ne présentent aucune ligne de stop. Ces quelques images méritent d'être labellisées une à une alors que les autres peuvent être traitées massivement.

#### Données d'entrée
En entrée, on a besoin d'un ensemble d'images. Le répertoire contenant ces images est à indiquer dans la variable ```targetDir```. Dans ce dossier, il faut aussi créer un fichier ```labels.json```.

Dans le cadre du projet, les images des courses sont numérotées et ont un format comme *XXX_cam-image_array_.jpg* où *XXX* est le numéro de l'image. Dans le cas où le suffixe est différent de *_cam-image_array_.jpg*, il faut modifier la variable ```nameWithoutPrefix```.

#### Données de sortie
En sortie, le fichier ```labels.json``` est complétée avecn pour chaque image, un label 0 ou 1.

Le fichier ```labels.json``` est donc une grosse avec des items du genre :

```python
{"img_name": "6210_cam_image_array_.jpg", "label_value": 0}
```

#### Fonctionnement

```cate.py``` s'utilise en commentant, décommentant certaines sections.

Pour l'utiliser efficacement, il faut parcourir les images du premier tour une à une à l'aide de son explorateur de fichiers. Cela donnera une idée du nombre d'images entre le moment où l'on ne voit plus la ligne et celui où on la revoit.

1. Sur la plage d'images où l'on est susceptible de voir la ligne, on décommente la section **LABELLISATION DES IMAGES UNE À UNE**, on commente la section **LABELLISATION MASSIVE DES IMAGES** et on lance le script. On est invité à se prononcer sur 20 images une à une dans l'invite de commande. S'il y a présence de ligne, il suffit de taper *y*, sinon de presser Entrée.

2. Sur la plage d'images où l'on n'est pas susceptible de voir la ligne, on commente la section **LABELLISATION DES IMAGES UNE À UNE** et on décommente la section **LABELLISATION MASSIVE DES IMAGES**. On ajuste la plage de la boucle (en conservant des multiples de 20 en bornes inférieue et supérieure) et on lance le script. Toutes les images passées sont par défaut considérées sans ligne.

3. On continue ainsi de suite jusqu'à avoir parcouru l'ensemble des images.

## Stop par Intelligence Artificielle
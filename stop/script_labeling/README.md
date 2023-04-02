# Labellisation des images

## ```cate.py```

Le fichier ```cate.py``` a été créé pour labelliser facilement les images d'une course. Le constat est le suivant. Lors d'une course, il existe seulement quelques images par tour où l'on aperçoit la ligne et toutes les autres ne présentent aucune ligne de stop. Ces quelques images méritent d'être labellisées une à une alors que les autres peuvent être traitées massivement.

### Données d'entrée
En entrée, on a besoin d'un ensemble d'images. Le répertoire contenant ces images est à indiquer dans la variable ```targetDir```. Dans ce dossier, il faut aussi créer un fichier ```labels.json```.

Dans le cadre du projet, les images des courses sont numérotées et ont un format comme *XXX_cam-image_array_.jpg* où *XXX* est le numéro de l'image. Dans le cas où le suffixe est différent de *_cam-image_array_.jpg*, il faut modifier la variable ```nameWithoutPrefix```.

### Données de sortie
En sortie, le fichier ```labels.json``` est complétée avecn pour chaque image, un label 0 ou 1.

Le fichier ```labels.json``` est donc une grosse avec des items du genre :

```python
{"img_name": "6210_cam_image_array_.jpg", "label_value": 0}
```

### Fonctionnement

```cate.py``` s'utilise en commentant, décommentant certaines sections.

Pour l'utiliser efficacement, il faut parcourir les images du premier tour une à une à l'aide de son explorateur de fichiers. Cela donnera une idée du nombre d'images entre le moment où l'on ne voit plus la ligne et celui où on la revoit.

1. Sur la plage d'images où l'on est susceptible de voir la ligne, on décommente la section **LABELLISATION DES IMAGES UNE À UNE**, on commente la section **LABELLISATION MASSIVE DES IMAGES** et on lance le script. On est invité à se prononcer sur 20 images une à une dans l'invite de commande. S'il y a présence de ligne, il suffit de taper *y*, sinon de presser Entrée.

2. Sur la plage d'images où l'on n'est pas susceptible de voir la ligne, on commente la section **LABELLISATION DES IMAGES UNE À UNE** et on décommente la section **LABELLISATION MASSIVE DES IMAGES**. On ajuste la plage de la boucle (en conservant des multiples de 20 en bornes inférieue et supérieure) et on lance le script. Toutes les images passées sont par défaut considérées sans ligne.

3. On continue ainsi de suite jusqu'à avoir parcouru l'ensemble des images.

## ```create_test_n_train.py```

Le fichier ```create_test_n_train.py``` permet de fusionner plusieurs datasets en un gros dataset qui est divisé en 2 afin d'avoir des images pour l'entrainement et d'autres pour le test.

### Données d'entrée
Pour utiliser le fichier, il faut ajuster quelques variables :

- ```datasetsDir``` : chemin du répertoire qui contiendra les répertoires des datasets d'entrainement et de test
- ```datasets``` : liste des répertoires contenant les images labellisées
- ```dirs``` : dossier à créer (pas besoin de modifier)
- ```moveDatasets```: indication de quels datasets serviront pour le train et quels datasets serviront pour le test


### Données de sortie
Dans le répertoire indiqué en ```datasetsDir```, des dossiers seront créés conformément aux noms indiqués dans ```dirs```. Ces dossiers seront chacun composés de 2 sous-dossier, un nommé *class_0* dans lequel se trouveront les images labellisées par un 1 (cf. ```cate.py```) et un autre nommé *class_1* dans lequel se trouveront les images labellisées par un 0 (cf. ```cate.py```).

Un json similaire à celui de ```cate.py``` sera créé dans le répertoire de ```datasetsDir```.

### Fonctionnement
Une fois les variables introduites dans la section *Données d'entrée* ajustées, il suffit d'exécuter le script.

## ```cropper.py```
Le fichier ```cropper.py``` sert simplement à rogner l'ensemble des images d'un dataset. Il peut être utile pour l'entrainement d'un modèle de ne conserver que la portion basse de l'image où la ligne est susceptible de se trouver.

> __Note__ : le fichier n'est pas récursif. Si le dataset est constitué en plusieurs dossiers, comme c'est le cas avec le script ```create_test_n_train.py```, il faut l'exécuter pour chaque sous-répertoire.

### Données d'entrée
Pour utiliser le fichier, il faut ajuster quelques variables :

- ```DIR``` : le chemin du répertoire qui contient les images
- ```CROP_PX```: nombre de pixels à rogner depuis le haut de l'image

### Données de sortie
Écrase les images du répertoire en les remplaçant par elles-mêmes rognées.

### Fonctionnement
Une fois les variables introduites dans la section *Données d'entrée* ajustées, il suffit d'exécuter le script.

## ```label-from-existing.py```
Le fichier ```label-from-existing.py``` sert à fusionner les répertoire *class_1* et *class_0* des datasets (comme on aurait en sortie de ```create_test_n_train.py```).

### Données d'entrée
Pour utiliser le fichier, il faut ajuster quelques variables :

- ```dossier_a_rapatrier``` : le chemin du répertoire qui contient les sous-répertoire *class_0* et *class_1*
- ```CROP_PX```: nombre de pixels à rogner depuis le haut de l'image

### Données de sortie
Copie les images des sous-répertoire *class_0* et *class_1* dans le répertoire parent.

### Fonctionnement
Une fois les variables introduites dans la section *Données d'entrée* ajustées, il suffit d'exécuter le script.
# Apprentissage supervisé

## Récupération des données de la course
Lors de la course, la voiture enregistre l'image, la direction et la vitesse a une fréquence définie dans des répertoires nommés tubs, stocké dans le répertoire ```data``` de la voiture.

Pour faciliter la récupération des données de la dernière course, nous conseillons de créer un tub par exécution de la voiture (sinon toutes les données sont stockées dans le même tub et il devient difficile de faire le tri entre les anciennes et les nouvelles données). Pour cela, dans le fichier ```cars/mycar/myconfig.py```, il suffit de décommenter la constante ```AUTO_CREATE_NEW_TUB``` et de la passer à True.

Pour entraîner le modèle, il est conseillé d’utiliser son PC et de ne pas réaliser l'entraînement sur la Raspberry pour des questions de puissance. Pour récupérer les données en vue de lancer un entraînement sur son ordinateur personnel, nous avons utilisé 2 méthodes :

- a. en utilisant la commande scp ou rsync entre macOS ou Linux et la Raspberry (ça n’avait pas l’air de marcher depuis Windows), 
- b. en créant un repository git distant

### a. Avec les commandes scp et rsync
La procédure est bien expliquée sur la [documentation](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/). 

La commande sous les conditions suivantes :

- La Raspberry est identifiée sur le réseau sous le nom imtaracing.
- Sur la Raspberry, les données sont stockées dans la “voiture” mycar.
- Sur mon ordinateur, j’ai créé une voiture mysim dans le répertoire ia_racing de mon dossier personnel.

est 

```
rsync -rv --progress --partial pi@imtaracing.local:~/projects/mycar/data/  ~/ia_racing/mysim/data/
```

### b. Avec git
Nous avons mis en place un repository git distant. Par conséquent, depuis la Raspberry on peut réaliser un push des données acquises vers le repository distant, puis nous pourrons réaliser un pull depuis notre terminal personnel.

Sur la Raspberry (via ssh) :
1. Réaliser un commit
    
    ```
    $ cd ~/projects/mycar/    
    ```

    ```
    ~/projects/mycar/$ git add .
    ```

    ```
    ~/projects/mycar/$ git commit -m “ajout courses XXXXX” 
    ```
 
2. Pusher sur le repo distant

    ```
    ~/projects/mycar/$ git push
    ```

Sur l’ordinateur :

1. Réaliser un push

    ```
    $ cd ~/ia_racing/repo-mycar/
    ```

    ```
    ~/ia_racing/repo-mycar/$ git pull
    ```

## Fusion de plusieurs tubs

Parfois, il peut être utile de fusionner plusieurs tubs entre eux (par exemple, on a très bien conduit sur plusieurs courses). Pour cela, nous avons créé le script ```remaster_data.py```.

Pour l'exécuter, il suffit de l'exécuter avec les paramètres suivants :

- -f le chemin du répertoire contenant tous les tubs qu'on souhaite fusionner
- -c le nombre de pixels qu'on souhaite rogner en haut de l'image
- -p les preprocessings qu'on souhaite utiliser
- -t le chemin du répertoire de destination dans lequel on veut que les tubs soient fusionnées et preprocesser

Un sous-répertoire par type de preprocessing indiqué en -p sera créé dans le répertoire indiqué en -t.

Par exemple, pour fusionner plusieurs tubs enregistrés dans le répertoire ```tubs_to_merge``` qu'on veut preprocesser avec les preprocessings 'bnw' et 'lines' rognés de 40px par rapport au haut de l'image dans le répertoire ```tub_master```, on peut taper :

```
python remaster_data.py -f ./tubs_to_merge -c 40 -p bnw lines -t tub_master
```

On aura alors dans le dossier ```tub_master``` 3 sous-répertoires :

1. Les images brutes
2. Les images rognées en noir et blanc (bnw)
3. Les images rognées et traitées avec le preprocessing lines

En outre, dans chaque répertoire on retrouvera les ```catalog_X.catalog```, ```catalog_X.catalog_manifest``` et les ```manifest.json``` comme dans un tub standard.

## Utiliser les modèles du framework donkeycar

### Création de nouveaux modèles
Le Framework intègre plusieurs modèles de base implémentés avec Keras (nous déconseillons Pytorch qui n’est pas toujours compatible avec Raspberry Pi OS).

Pour ajouter un nouveau modèle, la procédure est expliquée sur la [documentation du framework](https://docs.donkeycar.com/dev_guide/model/).


### Lancer un entraînement
Pour lancer un entraînement, il suffit de taper les commandes suivantes de se rendre dans le répertoire correspondant à la voiture qui contient les données.

```
$ cd ~/ia_racing/mysim/
```

Pour choisir le type de modèle qu'on souhaite entraîner, il faut modifier le fichier ```myconfig.py``` en tapant la commande suivante et modifier la constante ```DEFAULT_MODEL_TYPE```. Il peut être nécessaire de modifier les constantes ```IMAGE_W```, ```IMAGE_H``` et ```IMAGE_DEPTH``` si les images ont été préprocessées. D'autres paramètres peuvent influer sur l'entraînement comme la taille des lots d'images (```BATCH_SIZE```), le learning rate (```LEARNING_RATE```) et le nombre d'epochs (```MAX_EPOCHS```).

> __Note__ : le nombre d'epochs renseigné correspond au nombre maximum d'epochs qui pourront être utilisées pour l'entraînement car, par défaut, lorsque la valeur de la fonction de coût n'évolue plus, l'entraînement s'arrête (cela peut être désactivé en passant la constante ```USE_EARLY_STOP``` à False). Cet arrêt est intéressant pour éviter le surapprentissage.

```
~/ia_racing/mysim/$ nano myconfig.py
```

Pour finir et lancer l'entraînement, il suffit de taper la ligne suivante.

```
~/ia_racing/mysim/$ donkey train --tub ./data --model ./models/mypilot.h5
````

Lorsque le modèle est entraîné, il y a 2 modèles créés : un .h5 et un .tflite (à condition que la constante ```CREATE_TF_LITE``` de ```myconfig.py``` ait la valeur Vrai). On conseille d’utiliser le tflite qui est plus léger et annoncé par Google être conçu pour les appareils embarqués. En outre, nous avons rencontré des difficultés à exécuter les .h5 issus des entraînements sur windows, mais pas pour les tflite.

> __Note__ : Si la commande ne génère pas de tflite, il faut s’assurer d’être sur la branche main du repository donkeycar. Pour cela :
>
> 1. ``` $ cd ~/ia_racing/donkeycar/```
> 2. ``` ~/ia_racing/donkeycar/$ git branch ```
>
>S'il y a 2 branches (main et *master) :
>
> 3. ``` ~/ia_racing/donkeycar/$ git checkout main ```
> 4. ``` ~/ia_racing/donkeycar/$ git pull ```
>
> En relançant un entraînement, le problème devrait être résolu.


## Envoyer le modèle sur la voiture

### a. Avec les commandes scp et rsync

Pour migrer le modèle sur la voiture, on peut utiliser la commande scp.

```
$ scp ~/ia_racing/mysim/models/mypilot.tflite pi@imtaracing.local:~/projects/mycar/models/
````

### b. Avec git

Pour migrer le modèle sur la voiture, on peut utiliser git en pushant le modèle sur le repo distant.

Sur l’ordinateur :

1. ``` ~/ia_racing/repo-mycar/$ git add ./models/mypilot.tflite ```
2. ``` ~/ia_racing/repo-mycar/$ git commit -m “Ajout du modele XXXXXXX” ```
3. ``` ~/ia_racing/repo-mycar/$ git push ```


## Créer sa part et son modèle

Nous avons essayé de créer notre propre modèle et notre propre part avant d'utiliser l'intégration des modèles au framework. Même si cela s'est conclu par un échec (modèle peu concluant au niveau de la direction), cela est un bon exercice pour prendre en main les problèmes d'apprentissage.

### Constitution du dataset
Avant de concevoir son modèle et l'entrainer, il faut constituer un dataset.

Nous n'avions pas la voiture à disposition jusqu'en janvier et avons réalisé ces scripts avant.
Nous nous sommes procurés des données de course sur simulateur en ligne pour consistuer le dataset (car ils conduisent mieux que nous). La structure des données était telle que pour chaque image, il existait un fichier ```record_XXX.json``` indiquant la vitesse et la direction associées à cette image. Dans le cas où ces fichiers n'existaient pas (par exemple quand on récupérait les données de nos propres courses), les données des angles et des vitesses étaient stockés dans des fichiers ```catalog_X.catalog```. Le format n'étant pas du json, une rapide manipulation manuelle était à faire (rajouter une virgule à la fin de chaque ligne et entourer l'ensemble des lignes par des crochets) en amont.

Nous avons écrit le script ```dataset_drive/label.py``` pour condenser tous ces petits fichiers de plusieurs sources en un gros fichier json.

#### ```dataset_drive/label.py```

##### Données d'entrée

Les données d'entrée sont des constantes.

- ```TO_CLASSIFY_DIR``` = répertoire contenant des répertoire avec les images et les fichiers ```record.json``` associés à chaque images ou les fichiers ```catalog_X.catalog```
- ```LABELS_FILE_FILENAME``` = fichier dans lequel condenser toutes les données angulaires et de vitesse

##### Données de sortie

Après exécution du fichier, les images sont enregistrées dans le dossier parent du répertoire renseigné dans ```TO_CLASSIFY_DIR```. Si ces images étaient dans des sous-répertoires de ```TO_CLASSIFY_DIR```, alors leur nom a été préfixé par le nom du sous-répertoire.

Les donneés angulaires et de vitesse ont été condensées dans le fichier pointé par ```LABELS_FILE_FILENAME```.

### Création et inférence du modèle
Après avoir réfléchi au problème d'optimisation que l'on cherche à résoudre, on peut s'atteler à la création du modèle.

Le fichier permettant d'entraîner notre modèle (qui s'est révélé mauvais) est ```pilot_train.py```. Il peut réaliser un entraînement à partir des données présentes dans ```DATASET_DIR```. Le fichier ```pilot_test.py``` permet de faire l'inférence.
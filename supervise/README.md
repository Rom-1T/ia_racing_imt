# Apprentissage supervisé

## Récupération des données de la course
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

    1. ``` $ cd ~/ia_racing/donkeycar/```
    2. ``` $ ~/ia_racing/donkeycar/$ git branch ```

    S'il y a 2 branches (main et *master) :

    3. ``` $ ~/ia_racing/donkeycar/$ git checkout main ```
    4. ``` $ ~/ia_racing/donkeycar/$ git pull ```

    En relançant un entraînement, le problème devrait être résolu.


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

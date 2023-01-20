# Intégrer préprocessing

Pour intégrer le préprocessing, il y a 3 fichiers à modifier.

1. On ajoute une part au framework
2. On ajoute une config spécifique à la voiture
3. On ajoute une section dans l'exécution de la voiture


## 1. Créer de la part dans le framework donkeycar
Dans le framework donkeycar, copier le fichier *preprocessing.py* dans le répertoire *parts* de *donkeycar*.

## 2. Ajouter la config spécifique à la voiture
Dans la voiture courante, ajouter les lignes suivantes à la fin du fichier *myconfig.py* (et ajuster les paramètres) :

	PREPROCESSING = False
	PREPROCESSED_CROP_FROM_TOP = 0
	PREPROCESSED_METHODS = [] # ['gaussian', 'canny'], les preprocessings qu'on souhaite faire, dans l'ordre


## 3. Ajouter la part dans le script de la voiture

Dans la voiture courante, ajouter les lignes suivantes dans la méthode *drive* du fichier *manage.py* avant la part du drive supervisé (pour que l'input du drive supervisé soit bien l'image préprocessée).

	if cfg.PREPROCESSING:
        from donkeycar.parts.preprocessing import Preprocessing
        prepro = Preprocessing(cfg)
        V.add(prepro, inputs=['cam/image_array'], outputs=['cam/image_array'])

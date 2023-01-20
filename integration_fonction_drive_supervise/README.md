# Intégrer la conduite supervisée

Pour intégrer la conduite supervisée, il y a 4 fichiers à modifier.

1. On ajoute une part au framework
2. On ajoute le resnet entrainé
3. On ajoute une config spécifique à la voiture
4. On ajoute une section dans l'exécution de la voiture


## 1. Créer de la part dans le framework donkeycar
Dans le framework donkeycar, copier le fichier *supervised_driver.py* dans le répertoire *parts* de *donkeycar*.

## 2. Ajouter le modèle entrainé
Dans la voiture courante, copier le fichier *XXXXXXX.torch* dans le répertoire *models*.

## 3. Ajouter la config spécifique à la voiture
Dans la voiture courante, ajouter les lignes suivantes à la fin du fichier *myconfig.py* (et ajuster les paramètres) :

	SUPERVISED_DRIVER = True
	SUPERVISED_DEVICE = "cpu"
	SUPERVISED_STATE_DICT_PATH = os.getcwd() + "/models/XXXXXX.torch"
	SUPERVISED_THROTTLE_MAX = 0.5
	SUPERVISED_ANGLE_MAX = 1

et ajuster ce qui doit être modifié :

- le device d'exécution (ex : sur la GPU si besoin)
- *XXXXXX.torch* par le modèle choisi en 2.

## 4. Ajouter la part dans le script de la voiture

Dans la voiture courante, ajouter les lignes suivantes dans la méthode *drive* du fichier *manage.py* (dans mon cas, à la ligne 458) pour être après l'appel de la part *DriveMode*.

	if cfg.SUPERVISED_DRIVER:
        from donkeycar.parts.supervised_driver import SupervisedDrive
        s_driver = SupervisedDrive(cfg)
        V.add(s_driver, inputs=['cam/image_array'], outputs=['throttle', 'angle'])

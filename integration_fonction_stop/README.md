# Intégrer le lot stop

Pour intégrer le lot STOP, il y a 4 fichiers à modifier.

1. On ajoute une part au framework
2. On ajoute le resnet entrainé
3. On ajoute une config spécifique à la voiture
4. On ajoute une section dans l'exécution de la voiture


## 1. Créer de la part dans le framework donkeycar
Dans le framework donkeycar, copier le fichier *stop_detection.py* dans le répertoire *parts* de *donkeycar*.

## 2. Ajouter le modèle entrainé
Dans la voiture courante, copier le fichier *sigmoid-10e.pth* dans le répertoire *models*.

## 3. Ajouter la config spécifique à la voiture
Dans la voiture courante, ajouter les lignes suivantes à la fin du fichier *myconfig.py* (et ajuster les paramètres) :

	import os
	STOP_DETECTION = True # Active la part de détection de la ligne stop
	STOP_DETECTION_MODEL_PATH = os.getcwd() + "/models/sigmoid-10e.pth"  # Chemin du modele entraine
	STOP_DETECTION_DEVICE = "cpu" # device sur lequel on utilise le modele
	THROTTLE_STOP_DETECTION = 0.9 # la probabilité acceptable pour considérer qu'il y a une ligne
	LAP_COUNTER_MAX = 6 # Nombre de tours avant l'arret force du vehicule
	STOP_DETECTION_PREVIOUS_IMG_BASE = 3 # Nombre d'images avec ligne pour considerer qu'on a passe la ligne
	STOP_DETECTION_PRINT = True # Affichage des moments de detection de ligne


## 4. Ajouter la part dans le script de la voiture

Dans la voiture courante, ajouter les lignes suivantes à la fin de la méthode *drive* du fichier *manage.py* (dans mon cas, à la ligne 572).

	if cfg.STOP_DETECTION:
        from donkeycar.parts.stop_detection import StopDetection
        stop_detection = StopDetection(cfg)
        V.add(stop_detection, inputs=['cam/image_array', 'throttle'], outputs=['throttle', 'lap', 'end'])

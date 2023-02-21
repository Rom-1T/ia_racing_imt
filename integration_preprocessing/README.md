# Intégrer préprocessing

Pour intégrer le préprocessing, il y a plusieurs fichiers à modifier.

1. On ajoute une part au framework
2. On ajoute une config spécifique à la voiture
3. On ajoute la part preprocessing aux parts exécutées par la voiture
4. On envoie les images preprocessées au modèle de conduite
5. On modifie les parts en lien avec l'interface visuelle

La partie 6. rappelle comment lancer un modèle et permet de s'assurer que les changements ont correctement été effectués.

## Principe

Le preprocessing est réalisé par une part créée maison. Elle doit donc être intégrée au framework et dans les parts exécutées par la voiture.

Lors du développement du preprocessing, le choix a été fait de conserver l'image brute de la caméra intacte dans le « channel » ```cam/image_array```. Un autre « channel » ```prepro/image_cropped``` a été créé pour récupérer l'image brute rognée (qui peut servir à la partie stop). Un « channel » ```prepro/image_lines``` a également été créé pour récupérer l'image à l'issue du preprocessing complet. C'est donc ```prepro/image_lines``` qui sera à passer au modèle de conduite supervisée.

![](images/inputs-outpust-manage_py.png)

Dans ce document, nous verrons :

- comment intégrer le preprocessing dans donkeycar pour transformer les images acquises par la caméra
- comment les communiquer au modèle
- comment avoir un rendu visuel du preprocessing sur l'interface web

## 1. Créer de la part dans le framework donkeycar
Dans le framework donkeycar, copier le fichier ```preprocessing.py``` dans le répertoire ```donkeycar/parts```.

## 2. Ajouter la config spécifique à la voiture
Dans la voiture courante, ajouter les lignes suivantes à la fin du fichier *myconfig.py* (et ajuster les paramètres). Cela permettra d'activer le preprocessing et de régler le niveau de crop ainsi que la méthode de preprocessing. Bien sûr, lorsqu'on souhaite faire rouler la voiture en automatique, il faut que le modèle qui tourne ait été entrainé avec les mêmes paramètres de configuration, sinon cela créera une erreur.

	PREPROCESSING = True
	PREPROCESSED_CROP_FROM_TOP = 40
	PREPROCESSING_METHOD = "lines" # throttle150|throttle195|canny150|canny195|contour150|contour195|lines


## 3. Ajouter la part dans le script de la voiture

Afin d'exécuter le preprocessing sur les images, ajouter les lignes suivantes dans la méthode *drive* du fichier ```cars/mycar/manage.py```. Le placer après le bloc de condition ```if cfg.SHOW_FPS:``` semble être un endroit convenable.

	if cfg.PREPROCESSING:
        from donkeycar.parts.preprocessing import Preprocessing
        prepro = Preprocessing(cfg)
        V.add(prepro, inputs=['cam/image_array'], outputs=['cam/image_array'])

## 4. Envoyer les images preprocessées au modèle

L'envoi des données au modèle n'est probablement pas rédigé optimalement dans ```cars/mycar/manage.py```, mais voici une proposition qui fonctionne. Inititalement, on retrouve ce bloc de code dans ```manage.py``` :

	if cfg.TRAIN_BEHAVIORS:
		bh = BehaviorPart(cfg.BEHAVIOR_LIST)
		V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
		try:
			ctr.set_button_down_trigger('L1', bh.increment_state)
		except:
			pass

		inputs = ['cam/image_array', "behavior/one_hot_state_array"]

	elif cfg.USE_LIDAR:
		inputs = ['cam/image_array', 'lidar/dist_array']

	elif cfg.HAVE_ODOM:
		inputs = ['cam/image_array', 'enc/speed']

	elif model_type == "imu":
		assert cfg.HAVE_IMU, 'Missing imu parameter in config'
		# Run the pilot if the mode is not user.
		inputs = ['cam/image_array',
				'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
				'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
	else:
		inputs = ['cam/image_array']

On peut le modifier en changeant l'input dans le *else*, tel que présenté ci-dessous (notons que « … » indique une partie de code à ne pas changer) :

	if cfg.TRAIN_BEHAVIORS:
		…
	else:
		if cfg.PREPROCESSING:
			inputs = ['prepro/image_lines']
		else:
			inputs = ['cam/image_array']

## 5. Modifier les parts en lien avec l'interface visuelle

### 5.1. Passer l'image préprocessée en entrée de l'interface graphique

Dans ```cars/mycar/manage.py```, modifier la définition de *add\_user\_controller* en changeant les paramètres. Initialement, on a :

	def add_user_controller(V, cfg, use_joystick, input_image='cam/image_array'):

Rajouter un paramètre *input\_prepro\_image* :

	def add_user_controller(V, cfg, use_joystick, input_image='cam/image_array', input_prepro_image = 'prepro/image_lines'):

Dans ```cars/mycar/manage.py```, modifier les entrées et les sorties de *LocalWebController*. Initialement, on a :

	ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(ctr,
          inputs=[input_image, 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)

Modifier ce bloc de code par :

	ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(ctr,
          inputs=[input_image, input_prepro_image, 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'web/buttons', 'stopCount'],
          threaded=True)


Dans ```donkeycar/donkeycar/web_controller/web.py```, modifier les paramètres de la méthode *run* et *run\_threaded*. Pour *run*, on a initialement :

	def run(self, img_arr=None, num_records=0, mode=None, recording=None):
        return self.run_threaded(img_arr, num_records, mode, recording)

qu'il faut modifier en :

	def run(self, img_arr=None, img_prepro = None, num_records=0, mode=None, recording=None):
        return self.run_threaded(img_arr, img_prepro, num_records, mode, recording)

Pour *run\_threaded*, on a initialement :

	def run_threaded(self, img_arr=None, num_records=0, mode=None, recording=None):

qu'il faut modifier en :

	def run_threaded(self, img_arr=None, img_prepro = None, num_records=0, mode=None, recording=None):


### 5.2. Générer l'image préprocessée

Pour l'instant, l'image préprocessée est un array, on ne peut donc pas la visualiser. La classe *VideoPreproAPI* aura pour rôle de la générer et de rendre l'affichage disponible. En définissant correctement les handleurs de *LocalWebController* on pourra la récupérer à une URL donnée.

Dans la méthode *run\_threaded*, ajouter ```self.img_prepro = img_prepro``` à la ligne après ```self.img_arr = img_arr```.

Dans le fichier ```donkeycar/donkeycar/web_controller/web.py```, ajouter la classe suivante :

	class VideoPreproAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle after preprocessing.
    '''

    async def get(self):

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:

            interval = .01
            if served_image_timestamp + interval < time.time() and \
                    hasattr(self.application, 'img_prepro'):
                img = utils.arr_to_binary(self.application.img_prepro)
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)

Enfin, ajouter l'entrée suivante dans les *handlers* de la méthode *\_\_init\_\_* de la class *LocalWebController* :
	
	(r"/video_prepro", VideoPreproAPI)

ce qui donne :

	handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/wsCalibrate", WebSocketCalibrateAPI),
            (r"/calibrate", CalibrateHandler),
            (r"/video", VideoAPI),
            (r"/video_prepro", VideoPreproAPI),
            (r"/wsTest", WsTest),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

On peut donc désormais récupérer l'image à l'adresse http://localhost:8887/video_prepro

### 5.3. Modifier l'interface web

Pour afficher l'image sur l'interface web, il suffit de rajouter une balise img qui charge l'image à l'adresse http://localhost:8887/video_prepro dans le fichier ```donkeycar/donkeycar/web_controller/templates/vehicle.html```

	<img src="/video_prepro" />

Pour faciliter la procédure, on peut simplement remplacer le fichier ```donkeycar/donkeycar/web_controller/templates/vehicle.html``` par ```vehicle.html``` qui se trouve dans le dossier ```web``` du répertoire de ce document. *Note : On peut faire même avec le fichier ````base.html```.

## 6. Lancement du modèle et vérification des changements

### 6.1. Lancer un entrainement avec des images preprocessées

Tel que le preprocessig a été intégré, il n'enregistre pas les images preprocessées (pour conserver les images brutes et en faire par la suite des usages multiples). Pour récupérer les images brutes de la voiture physique (Raspberry pi), il faut exécuter la ligne de code suivante :

	rsync -rv --progress --partial pi@imtaracing.local:~/mycar/data/  ~/mycar/data/

Source : [Documentation donkeycar](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/)

Ensuite, une fois les images récupérées, on peut générer les images préprocessées en exécutant le script ```preprocessing/generate_preprocessing_datasets_images.py``` qui se trouve sur le [GitHub *ia\_racing\_imt* sur la branche *drive_supervise*](https://github.com/Rom-1T/ia_racing_imt/tree/drive_supervise) en décommentant les sections nécessaires.

En déplaçant les images générées dans un dossier ```images``` au même niveau que les *catalogs* de la conduite, il est possible de lancer un entrainement sur les images préprocessées avec la commande (dans le cas présent, les catalogs et le dossier images se trouvent dans la voiture courante) :

	~\mycar$ donkey train --tub ./data --model ./models/mypilot.h5

Source : [Documentation donkeycar](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/)

On peut alors exécuter le modèle de conduite automatique supervisée en exécutant la commande suivante :

	python manage.py drive --model ~/mycar/models/mypilot.h5

Source : [Documentation donkeycar](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/)

Il faut au préalable s'assurer d'avoir paramétré de la même manière l'entrainement et ```cars/mycar/myconfig``` pour ne pas avoir d'erreur (notamment au niveau du crop). Un nommage de fichier commun pour l'entrainement du modèle peut faciliter le paramétrage du preprocessing pour la course.

Si on souhaite migrer le modèle dans la voiture, sur la Raspberry, il faudra au préalable téléverser le modèle sur la Raspberry grâce à la commande suivante :

	rsync -rv --progress --partial ~/mycar/models/ pi@imtaracing.local:~/mycar/models/

Source : [Documentation donkeycar](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/)
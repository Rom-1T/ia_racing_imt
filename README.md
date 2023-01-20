# ia\_racing\_imt
## Drive supervise
### Pour enregistrer un nouveau dataset
Les datasets sont dans le répertoire *supervise/dataset_drive*.

Toutes les images sont du train. Il n'y a donc pas de fichier test.

Pour classer les images du dataset, on peut utiliser le script : 
	
	supervise/dataset_drive/label.py

Il va récupérer les dossier d'images  de *supervise/dataset\_drive/a_classer* et ajouter les images dans *supervise/dataset_drive* et ajouter les throttle + angle associés à chaque image dans *labels.json*.

/!\ Il peut y avoir des modifications à faire selon la manière dont sont enregistrées les throttles + angles des datasets (ex : dans plusieurs fichiers, dans un seul…)

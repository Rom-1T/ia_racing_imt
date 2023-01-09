# ia\_racing\_imt
## Stop
### Pour enregistrer un nouveau dataset
Les datasets sont dans le répertoire stop/datasets

Pour classer les images du dataset, on peut utiliser le script : 
	
	cate.py

* 1  si l'image a une ligne de stop
* 0 sinon

### Créer des datasets train/test
Pour créer un dataset train et un dataset test, on peut utiliser le script 

	create_test_n_train.py

Il créer des dossiers train et test avec des répertoires class_0 et class_1 dedans.

### Algo de classification
* Resenet 18

	stop/resnet18.py


## Intégration de la fonction stop

Voir fichier integration_fonction_stop

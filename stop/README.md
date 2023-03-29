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

## Stop par Intelligence Artificielle

### Formalisation
La formalisation du problème d'optimisation est explicitée dans le document ```Formalisation.pdf```.

### ```stop-ia.py```

#### Prérequis
Installation de PyTorch.

#### Données d'entrée
Les données d'entrée sont à renseigner à travers la définition de constantes :

- ```ROOT_DIR``` : Fichier racine du projet
- ```MODELE_ENREGISTRE``` : Indique s'il faut charger un modèle déjà existant
- ```ENREGISTRER_MODELE``` : Indique s'il faut enregistrer le modèle
- ```MODELE_NAME``` : Indique le nom du modèle à charger et/ou enregistrer
- ```N_EPOCHS``` : Indique le nombre d'epochs à parcourir
- ```BATCH_SIZE_TRAIN``` : Indique la taille des lots pour l'entrainement
- ```BATCH_SIZE_VALIDATE``` : Indique la taille des lots pour la validation

THRESHOLD : Indique le seuil de confiance que l'on souhaite pour considérer la présence d'une ligne

#### Données de sortie

En fonction des choix faits, les données de sortie peuvent être :
- le fichier du modèle
- les statistiques (de perte en fonction du seuil de confiance)

#### Fonctionnement

Le script est constitué de 3 classes :

##### La classe StopLineDataset
La classe StopLineDataset sert à créer les datasets au bon format pour pouvoir être utilisé par PyTorch. Elle permettra par la suite de pouvoir créer un DataLoader.

##### La classe Classif
C'est la classe maîtresse du fichier. Elle permet de :

- charger le modèle (méthode ```load_model```)
- charger les datasets (méthode ```set_dataset```)
- entraîner le modèle (méthode ```train```)
- vérifier les performances du modèle (validation avec la méthode ```validate```)
- afficher les statistiques (loss en fonction du nombre d'epochs avec les méthodes ```show_stats``` et ```test_several_thresholds```)

##### La classe Decision
C'est la classe qui permet d'implémenter le critère de décision sur la présence ou non de la ligne. En l'état, elle sert simplement de comparaison de la probabilité d'avoir une ligne renvoyée par le modèle à une seuil (threshold) défini.

##### Le main

Dans le main, on initie le modèle.

```python
if not(MODELE_ENREGISTRE):
    m = models.resnet18(pretrained=True)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 1),
        nn.Sigmoid()
    )
else:
    m = ROOT_DIR + MODELE_NAME

c = Classif()
c.load_model(m)
```

On indique ensuite la fonction de coût et l'optimiser.

```python
c.set_criterion(nn.BCELoss())
c.set_optimizer(optim.SGD(c.model.parameters(), lr=0.001, momentum=0.9))
c.scheduler = lr_scheduler.StepLR(c.optimizer, step_size=7, gamma=0.1)
```

On ajoute des datasets en leur définissant des noms.

```python
c.set_dataset('TRAIN', ROOT_DIR + 'dataset_sigma_crop/train', data_batch_size=BATCH_SIZE_TRAIN)
c.set_dataset('TEST', ROOT_DIR + 'dataset_sigma_crop/test', data_batch_size=BATCH_SIZE_TRAIN)

c.set_dataset('VALIDATION', ROOT_DIR + 'validation_dataset_sigma_crop', data_batch_size=BATCH_SIZE_VALIDATE)
```

On lance un entraînement sur les datasets portant les noms TRAIN (avec retropropagation) et TEST (sans retropropagation).

```python
c.train('TRAIN', 'TEST', N_EPOCHS)
```

On affiche les statistiques pour plusieurs seuils de confiance.

```python
c.test_several_thresholds([0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99])
```

On enregistre le modèle

```python
c.save(ROOT_DIR + MODELE_NAME)
```

On vérifie les performances sur un dataset indépendant (ici appelé VALIDATION)

```python
r, n, s, e = c.validate('VALIDATION')

print("Nombre d'images : ", n['total'])
print()
print("Détection de l'absence de ligne :")
print("   Erreurs :", n['errors']['should_be_0'])
print("   Réussites :", n['successes']['is_0'])
print()
print("Détection de la présence de ligne :")
print("   Erreurs :", n['errors']['should_be_1'])
print("   Réussites :", n['successes']['is_1'])

print(e) # Nombre d'erreurs au total
```
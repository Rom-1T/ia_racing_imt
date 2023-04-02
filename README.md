# IA Racing

Bienvenue sur le projet *IA Racing* d'IMT Atlantique. Dans ce repository, vous trouverez l'intégralité de nos avancées sur ce projet.

Les élèves ayant travaillé sur ce projet en 2022-2023 sont :

| Élève                 | TAF       |
| ---                   | ---       |
| Romain TESSIER        | ASCY      |
| Pierre GAILLARD       | LOGIN     |
| Wyatt MARIN           | ROBIN     |
| Malick BA             | ASCY      |
| Jhon MUÑOZ            | LOGIN     |
| Amaury COLIN          | COPSI     |

## Structure du repository

Le contexte de la course introduit 3 grands temps : démarrer (START), rouler (DRIVE), et s'arrêter (STOP). Nous avons divisé les étapes en 3 lots : le lot START, le lot STOP, et le lot DRIVE (qui est en réalité un lot double car nous nous sommes divisés en 2 équipes sur ce lot, avec des stratégies différentes).

Ainsi, la structure du repository est telle que :

- répertoire start : lot START (code de détection d'un feu vert et explications)
- répertoire stop : lot STOP (explications de 2 stratégies de détection (IA et déterministe) de la ligne de stop, scripts de labelling, et codes)
- drive_renforcement : lot DRIVE en stratégie d'apprentissage par renforcement (explications de Gym Environnement et de l'autoencoder, sources, et codes)
- drive_supervise : lot DRIVE en stratégie d'apprentissage supervisé (explications, process, et code)

En addition, nous avons inclus 2 répertoires supplémentaires :
- simulateur : comment modifier le simulateur pour le faire correspondre aux besoins
- integration : comment monter la voiture, calibrer la caméra, intégrer le framework donkeycar, ajouter de nouvelles parts, lancer la voiture…

```
*
|
|-- start
     |-- Explication du lot START
     |-- Code du lot START
|
|-- stop
     |-- Explication du lot STOP
     |-- Formalisation du problème d'optimisation
     |-- Code détection déterministe
     |-- Code détection par IA
     |-- Datasets
          |-- Images simulateurs
          |-- Images réelles
     |-- script_labelling
          |-- Explication du fonctionnement des scripts d'aide au labelling des images
          |-- Scripts d'aide au labelling des images
     |-- settings
          |-- Explication du fonctionnement du script d'aide à la recherche de plages de couleur
          |-- Script d'aide à la recherche de plages de couleur
|
|-- supervise
     |-- Explication du DRIVE supervisé
     |-- Codes de création et entraînement d'un modèle maison
     |-- Constitution d'un dataset
     |-- Fusion de tubs
|
|-- drive_renforcement
     |-- Explication du DRIVE par renforcement
     |-- Framework du DRIVE par renforcement
|
|-- simulateur
     |-- Explication des modifications réalisables sur le simulateur
|
|-- integration
     |-- Explication de l'installation de Raspbian Os Lite, Donkeycar, Tensorflow, Pytorch et Cuda
     |-- calibration_camera
          |-- Explication de la calibration pour retirer l'effet fish-eye
          |-- Codes de calibration
     |-- framework_donkeycar_parts
          |-- Explication du fonctionnement par parts du framework Doonkeycar
          |-- parts
               |-- Explication intégration de nos parts
               |-- Codes de nos parts
     |-- montage_voiture
          |-- Explication du montage de la voiture
     |-- mycar
          |-- Explication du fonctionnement de l'application donkeycar (aka voiture)
          |-- Fichiers de la voiture (manage.py, myconfig.py…)
     |-- preprocessing
          |-- Explication du preprocessings et aperçus
          |-- Scripts utiles
        
```

### Liens par lots

- Lot START
    - [Explication du lot START](https://github.com/Rom-1T/ia_racing_imt/blob/main/start)
    - [Intégration de la part START](https://github.com/Rom-1T/ia_racing_imt/blob/main/integration/framework_donkeycar_parts/parts)
- Lot STOP
    - [Explication du lot STOP](https://github.com/Rom-1T/ia_racing_imt/tree/main/stop)
    - [Explication du fonctionnement des scripts d'aide au labelling des images](https://github.com/Rom-1T/ia_racing_imt/tree/main/stop/script_labeling)
    - [Explication du fonctionnement du script d'aide à la recherche de plages de couleur](https://github.com/Rom-1T/ia_racing_imt/tree/main/stop/settings)
    - [Intégration de la part STOP](https://github.com/Rom-1T/ia_racing_imt/blob/main/integration/framework_donkeycar_parts/parts)
- Lot DRIVE
    - DRIVE supervisé
        - [Explication du DRIVE supervisé](https://github.com/Rom-1T/ia_racing_imt/tree/main/supervise)
        - [Installer Tensorflow et Cuda](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration)
        - [Comprendre les données enregistrées par la voiture](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/mycar)
        - [Mise en place d'un dataset](https://github.com/Rom-1T/ia_racing_imt/tree/main/supervise)
    - DRIVE par renforcement
        - [Ajuster le simulateur](https://github.com/Rom-1T/ia_racing_imt/tree/main/simulateur) et [ajouter un circuit](https://github.com/Rom-1T/ia_racing_imt/tree/main/drive_renforcement)
        - [Intégrer et utiliser le framework RL-Baseline3-Zoo framework](https://github.com/Rom-1T/ia_racing_imt/tree/main/drive_renforcement)
        - [Installer Pytorch sur la voiture](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration)
- Intégration
    - [Monter la voiture + coupe-circuit](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/montage_voiture)
    - [Installer l'OS](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration)
    - Donkeycar
        - [Installer Donkeycar](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration)
        - [Explication du fonctionnement des parts](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/framework_donkeycar_parts)
        - [Intégrer nos parts](https://github.com/Rom-1T/ia_racing_imt/blob/main/integration/framework_donkeycar_parts/parts)
        - [Mettre en place du preprocessing](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/preprocessing) et l'[intégrer](https://github.com/Rom-1T/ia_racing_imt/blob/main/integration/framework_donkeycar_parts/parts)
        - [Créer et paramétrer sa voiture](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/mycar)


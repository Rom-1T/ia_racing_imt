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
- integration : comment monter la voiture, intégrer le framework donkeycar, ajouter de nouvelles parts, lancer la voiture…

*
|
|-- start
     | ---- [Structure du code du lot START](https://github.com/Rom-1T/ia_racing_imt/blob/main/start/README.md)
|
|
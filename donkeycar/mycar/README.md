# ia_racing_imt
Ce dossier contient les fichiers à modifier/placer dans sa simulation de véhicule pour pouvoir utiliser des modèles pytorch avec manage.py drive.
## manage
Ce fichier contient l'instanciation du véhicule avec l'attribution de parts qui seront activés à la fréquence choisie (de base 20Hz). 
- Une part pytorch.py a été rajouté, le code de manage.py a donc été adapté. 
## models/pytorch
Ce dossier doit contenir le modèle drive.zip (et potentiellement le modèle ae.pkl). Ils seront utilisés en précisant les arguments:
- --type pyrorch
- --path /mycar/models/pytorch/

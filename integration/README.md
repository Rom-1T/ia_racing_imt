# Integration

## Install Raspbian Os Lite

Here are the steps to install Raspbian OS Lite on a fresh Raspberry Pi 4B:

1. Download the Raspbian OS Lite image from the official Raspberry Pi website. Make sure to choose the appropriate version for your Raspberry Pi model.
2. Flash the Raspbian OS Lite image onto a microSD card using a tool like Raspberry Pi Imager or Etcher.
3. Insert the microSD card into the Raspberry Pi 4B and connect the power supply, keyboard, mouse, and monitor.
4. Wait for the Raspberry Pi to boot up and login using the default username and password. The default username is "pi" and the default password is "raspberry".
5. Configure the network settings. If you are using a wired connection, you should already be connected to the network. If you are using a wireless connection, you need to configure the network settings using the "wpa\_supplicant.conf" file. :

Run the following command to edit the "wpa\_supplicant.conf" file:

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

Add the following lines to the bottom of the file, replacing "YOUR\_SSID" and "YOUR\_PASSWORD" with your actual SSID and password:

```makefile
network={
    ssid="YOUR_SSID"
    psk="YOUR_PASSWORD"
}
```

Now reboot the rasp.

7. (Optional) Configure the timezone and keyboard settings. You can configure the timezone using the "raspi-config"
8. Enable SSH access (If not already done in the installation) :

Run the following command to open the "raspi-config" tool:

* ```lua
  sudo raspi-config
  ```
* Use the arrow keys to navigate to "Interfacing Options" and then "SSH". Follow the on-screen instructions to enable SSH access.
* Save the changes and exit the tool.
* Restart the SSH service using the following command:

```
sudo systemctl restart ssh
```

That's it! You now have a fresh installation of Raspbian OS Lite on your Raspberry Pi 4.

## Connect to the Raspberry

To connect to the Raspberry, you may use ssh protocol. Ssh is an easy way to get access to the Raspberry Pi from your computer as if you were using the Raspberry directly connected to a screen. Therefore, you can use the terminal in the same way.

Here are the steps to authenticate yourself in ssh:

1. Switch on the Raspberry Pi.
2. You need to be connected on the same network as you Raspberry is. If there is no wifi set, then you can connect your computer to the Raspberry by mean of an ethernet wire.
3. Check if the Raspberry Pi is on the network by typing:

	```
	ping imtaracing.local
	```
	(here *imtaracing* is the name of our Raspberry).
	
4. Connect to the Raspberry in SSH with the access you previously defined by typing:

	```
	ssh <user>@<host>.local
	```
	
	in our case:
	
	```
	ssh pi@imtaracing.local
	```

## Installing Donkey Car and preparing the env

Go to [https://docs.donkeycar.com/guide/robot\_sbc/setup\_raspberry\_pi/](https://docs.donkeycar.com/guide/robot\_sbc/setup\_raspberry\_pi/)

## Installing PyTorch

Installing PyTorch on a Raspberry Pi 4 can be a little challenging. In our case, we downloaded a wheel version of Torch 1.7.0. It is clearly not the latest version of PyTorch, but it is the easiest way of installing PyToch.

Try : [https://github.com/Kashu7100/pytorch-armv7l](https://github.com/Kashu7100/pytorch-armv7l) or [https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B](https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B)

An other solution would have been to compile PyTorch from source. (HARD)

## Installing Tensorflow & CUDA

Tensorflow est la bibliothèque pemettrant l’utilisation des réseaux de neurones implémentés dans DonkeyCar et CUDA permet d’utiliser le GPU d’un ordinateur et ainsi réduire drastiquement les temps de calcul pour les entraînements (2 à 3 fois moins de temps observé). Il faut pour cela avoir une carte NVIDIA disponible.

L’installation est relativement laborieuse donc voici une explication de la procédure à réaliser pour installer ces deux modules.

**Tensorflow**

Donkeycar a été prévu pour fonctionner avec Tensorflow **2.2** mais des versions plus récentes peuvent être utilisées. Cependant, avec Windows, la version **2.10** de Tensorflow était la dernière version permettant l’utilisation du GPU:&#x20;

{% embed url="https://www.tensorflow.org/install/pip?hl=fr#windows-native" %}

Il faudra donc installer une version égale ou inférieure à la version **2.10** pour utiliser CUDA avec Windows.

On peut installer Tensorflow avec la commande:

`pip install tensorflow==<version>`

**CUDA**

En fonction de la version Tensorflow installée, il faudra installer la version de CUDA compatible. Ceci est disponible sur le site de Tensorflow:

[https://www.tensorflow.org/install/source\_windows?hl=fr](https://www.tensorflow.org/install/source\_windows?hl=fr)

Il faudra également installé tous les pilotes et bibliothèques décrits dans le guide d’installation de CUDA (voir configuration logicielle requise) :&#x20;

[https://www.tensorflow.org/install/gpu?hl=fr#windows\_setup](https://www.tensorflow.org/install/gpu?hl=fr#windows\_setup)

Si certaines bibliothèques ne sont pas prises en compte, il faudra ajouter leur chemin dans les variables du système (Panneau de configuration - Modifier les variables d'environnement  - Variables d’environnement) comme décrit dans le guide.

Pour savoir si l’installation de CUDA est effective il faudra prêter attention aux erreurs éventuelles affichées lors du lancement d’un entraînement et surveiller l’utilisation du GPU. Certaines bibliothèques supplémentaires peuvent être nécessaires comme la bibliothèque _zlib_.
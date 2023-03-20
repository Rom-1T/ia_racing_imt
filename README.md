# Integration

#### Install Raspbian Os Lite

Here are the steps to install Raspbian OS Lite on a fresh Raspberry Pi 4B:

1. Download the Raspbian OS Lite image from the official Raspberry Pi website. Make sure to choose the appropriate version for your Raspberry Pi model.
2. Flash the Raspbian OS Lite image onto a microSD card using a tool like Raspberry Pi Imager or Etcher.
3. Insert the microSD card into the Raspberry Pi 4B and connect the power supply, keyboard, mouse, and monitor.
4. Wait for the Raspberry Pi to boot up and login using the default username and password. The default username is "pi" and the default password is "raspberry".
5. Run the following command to update the system and packages:

```sql
sudo apt-get update && sudo apt-get upgrade
```

6. Configure the network settings. If you are using a wired connection, you should already be connected to the network. If you are using a wireless connection, you need to configure the network settings using the "wpa\_supplicant.conf" file. :&#x20;

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

#### Installing Donkey Car and preparing the env

Go to [https://docs.donkeycar.com/guide/robot\_sbc/setup\_raspberry\_pi/](https://docs.donkeycar.com/guide/robot\_sbc/setup\_raspberry\_pi/)

#### Installing PyTorch

Installing PyTorch on a Raspberry Pi 4 can be a little challenging. In our case, we downloaded a wheel version of Torch 1.7.0. It is clearly not the latest version of PyTorch, but it is the easiest way of installing PyToch.

Try : [https://github.com/Kashu7100/pytorch-armv7l](https://github.com/Kashu7100/pytorch-armv7l) or [https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B](https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B)

An other solution would have been to compile PyTorch from source. (HARD)

#### Create the parts

#### Integrate the model

#### Performance issues

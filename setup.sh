#!/bin/bash

# Mise à jour du système
echo "Mise à jour du système..."
sudo apt-get update
sudo apt-get upgrade -y

# Installation des dépendances système
echo "Installation des dépendances système..."
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential

# Création de l'environnement virtuel
echo "Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Mise à jour de pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installation des dépendances Python
echo "Installation des dépendances Python..."
pip install -r requirements.txt

echo "Configuration terminée !"
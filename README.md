# Single-Image-Super-Resolution

Repo du stage d'été de M1 (2018-2019) dans l'équipe MLIA (encadrant Patrick Gallinari)

## Carnet de bord
  - semaine du 27/05 (début du stage de Keyvan)
    - Lecture d'articles de recherche (GAN, GAN conditionnel, SRGAN, Amortized MAP Inference for image SR)
    - Téléchargement de celebA
    - Tuto DCGAN pytorch
    
  - semaine du 03/06
    - Implémentation de SRGAN
      - adverserial loss: Generator et Disciminator customisables
      - content loss via MaskedVGG (concaténation de VGG22, VG54, ...)
      - initialisation par défaut de PyTorch, on peut utiliser une MSE pendant 1 epoch comme initialisation
      ![Résultats SRGANx2](./results/x2.png)
    - Exécution du code sur GPU
    - Sauvegarde des réseaux entraînés
    - Lecture de AmbientGAN et de l'article d'Emmanuel
    - téléchargement de MNIST (base de données plus simple pour les modèles). Pas pratique au final car images trop petites (et un seul channel)
  
  - 10/06
    - Début de prise en main du [framework d'Emmanuel (privé)](https://github.com/emited/tmd_framework) qui se sert de:
      - [Ignite](https://pytorch.org/ignite/) (pour ne pas avoir à écrire la boucle d'entraînement et logger les résultats)
      - [Sacred](https://sacred.readthedocs.io/en/latest/index.html) (pour sauvegarder les configs utilisées) 
  
  - 17/06
    - Emmanuel nous présente son modèle de super résolution (qui se base sur AmbientGAN)
    - Implémentation du modèle en partant de notre code du SRGAN
    - Début d'implémentations de SRGAN et du modèle d'Emmanuel sur le framework d'Emmanuel
  
  - 24/06
    - Téléchargement et entraînement sur la base de données flickr8k
    - Implémentation de SRGAN sur le framework d'Emmanuel
    - Tentatives (infructueuses) d'entraînement du réseau en x4:
      - La loss D est toujours très faible
      - La loss G augmente parfois énormément puis stagne pendant une centaine d'itérations puis redescent
    - Lecture d'articles et tutos sur les améliorations que l'on peut d'implémenter sur notre réseau 
  
  - 01/07
    - La dernière features map de G avant upsampling était de taille 64, elle passe à 256 comme dans l'article de SRGAN
    - Améliorations pour entraînement x4:
      - Initialisation de G en utilisant seulement une MSE pendant une epoch,
        puis entraînement du discriminateur uniquement (moins d'une epoch pour éviter le sur-apprentissage)
      - real_label = .9 (au lieu de 1) pour que D ne soit pas sûr de lui
      - Sauvegarde des anciennes images générées pour réentrainer D dessus
      - Normalisation des gradients à 1 (c'est en fait mieux sans)
      - spectralNorm dans G et D
      - features map pour la content loss extraites avant activation
    - Le calcul de lr par interpolation de hr, on dépasse un peu de [-1, 1]. On rescale les images dans [-1, 1] si elles dépassent de l'intervalle.
    - En partant d'un mếme réseau (avec un dataset non mélangé), deux entraînements donnent des images complètement différentes,
      on peut mettre torch.backends.cudnn.deterministic = True et torch.backends.cudnn.benchmark = False
  
  - 08/07
    - Augmentation graduelle du poids de la loss adversaire (par rapport à la content loss) tant que cela améliore le rendu.
    
    - Création d'un script de visualisation des résultas
      ![Résultats SRGANx4](./results/x4_e-2_2epoch.png)
      (1ère ligne : LR SR HR UR 2ème ligne : interpolation bicubique de l'image d'au dessus)
  
    - Lecture de ESRGAN. Améliorations potentielles :
      - architecture de G
      - relativistic D
      - enlever les BatchNorm
    
  
  - 15/07
    - Progressive Generator:
      - Implémenté via "load_state_dict(strict=False)". Il y a des bugs e.g. https://github.com/pytorch/pytorch/pull/22545.
      - Un réseau x4 entraîné à partir d'un réseau x2 est meilleur
      - On peut geler toutes les couches à part la convolution rajoutée afin d'aller plus vite
      - Les images de celeba sont trop petites pour le x8. La MSE est floue, le GAN invente un visage:
      <img src="./results/invente.png" width="100">
      ![Invente](./results/invente.png | width=100)
  
## todo
  - améliorations potentielles pour l'implémentation de SRGAN (résultats déjà excellents)
    - poids de la content loss
    - lr_decay
    - télécharger la même base de données que dans l'article pour pouvoir comparer
    - ajouter des tests
    
  
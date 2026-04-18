Datasets - Epilepsy

CHB-MIT n=23
23 pediatric subjects with intractable seizures. (5 males, ages 3–22; and 17 females, ages 1.5–19; 1 n/a)
https://zenodo.org/records/10259996




CEBRA
→ espace latent: plus généralisable, interprétable 
→ médecine: on veut aussi comprendre ce qu’il se passe → explainable AI 




Définir nos hypothèses (prédire en t+1 ou en t) 
Preprocess low band-high band artefact bruit  
dim = 18*4h
Train CEBRAs → test dim espace latent
entraîne un modèle pour prédire KNN ML
Accuracy: cross-validation/(LOSO?) 
Tests de validation: permutation, cross-entropy entre espace latents 
Explainability 
interprétation + solution possible ( partie du medecin )
présentation 

Prédiction en temps réel simulé (sliding window, potentiellement feed par les résultats de xAI) 
Animations → point qui bouge au fur et à mesure dans l’espace latent
pour prédictions: EEG qui défile avec la prédiction du modèle qui s’actualise

Côté pédiatrie > éviter d’ouvrir le crâne pour iEEG tout en comprenant d’où peut être la cause de la crise 

Étudier Transitions de phase dans l’espace latent entre normal et seizure 

Vérifier que modèle capture bien phases et pas individus personnels -> colorer par individu 

Pb -> si foyers sont différents selon individus = espaces latents incomparables -> features de réseau ?
Transition vers état ictal: transition de phase caracterisable par marqueurs précoces dans ce réseau? -> prédire les crises, -> localiser le foyer de manière non invasive 

xCEBRA


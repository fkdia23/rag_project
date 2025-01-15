Pour garantir une exécution transparente de vos scripts, et compte tenu du fait que certaines fonctions de ces scripts reposent sur des bibliothèques externes, il est essentiel d'installer certaines bibliothèques prérequises avant de commencer. Pour ce projet, les bibliothèques clés dont vous aurez besoin sont Gradio pour créer des interfaces web conviviales et IBM watsonx AI pour exploiter des modèles LLM avancés à partir de l'API d'IBM watsonx.

    gradio vous permet de créer rapidement des applications web interactives, rendant vos modèles d'IA facilement accessibles aux utilisateurs.
    ibm-watsonx-ai pour utiliser les LLM de l'API watsonx.ai d'IBM.
    langchain, langchain-ibm, langchain-community pour utiliser les fonctionnalités pertinentes de Langchain.
    chromadb pour utiliser la base de données chroma comme base de données vectorielle.
    pypdf est nécessaire pour charger des documents PDF.


1- chmod +x setup.sh run.sh
2- pip freeze > requirements.txt
3- sudo apt install nvidia-cuda-toolkit

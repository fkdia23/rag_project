from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch
import gradio as gr
import warnings

# Désactiver les warnings inutiles
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

## Initialisation du LLM
def get_llm():
    """
    Initialise et retourne le modèle de langage
    """
    model_id = "unsloth/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Création du pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.95,
        max_length=2048,  # Permet des réponses longues
        pad_token_id=tokenizer.eos_token_id
    )
    
    # LangChain wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

## Chargement des documents PDF
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Découpage des textes
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Base vectorielle
def vector_database(chunks):
    embedding_model = get_embeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Modèle d'embedding
def get_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
        
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

## Création du retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## Nettoyage et post-traitement de la réponse
def clean_response(response):
    """
    Nettoie la réponse pour s'assurer qu'elle est complète et propre.
    """
    response = response.strip()
    if not response:
        return "Je n'ai pas pu trouver une réponse dans le document."
    return response

## Chaîne de QA avec Retrieval
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    
    # Configurer RetrievalQA pour retourner uniquement la réponse
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False  # Supprime les métadonnées
    )
    
    # Obtenir la réponse brute
    result = qa.run(query)
    
    # Nettoyer et retourner la réponse
    return clean_response(result)

## Interface Gradio
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Télécharger un fichier PDF", file_count="single", file_types=['.pdf'], type="filepath"),  # Upload de fichier
        gr.Textbox(label="Votre question", lines=2, placeholder="Tapez votre question ici...")
    ],
    outputs=gr.Textbox(label="Réponse"),
    title="Chatbot RAG",
    description="Chargez un document PDF et posez des questions. Le chatbot répondra en se basant sur le document fourni."
)

# Lancer l'application
rag_application.launch(server_name="0.0.0.0", server_port=7861)

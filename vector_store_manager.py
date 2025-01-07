from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama.embeddings import OllamaEmbeddings
import logging
from typing import List
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorStoreManager:
    """
    Manages the creation and retrieval of the vector store.
    
    Parameters:
        model (str): Name of the embedding model to use.
    """
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = model

    def create_vector_store(self, text_chunks: List[Document], store_path: str = "faiss_index"):
        """
        Create a vector store and save it locally.
        
        Parameters:
            text_chunks (List[str]): List of text chunks to store.
            store_path (str): Path where the FAISS index will be saved.
        """
        # embeddings = OllamaEmbeddings(model=self.model)
        # Extract the text content from Document objects
        text_content = [doc.page_content for doc in text_chunks]
        embeddings=HuggingFaceEmbeddings(model_name=self.model)
        vector_store = FAISS.from_texts(text_content, embedding=embeddings)
        vector_store.save_local(store_path)
        logging.info(f"Vector store created and saved to '{store_path}'.")

    def load_vector_store(self, store_path: str = "faiss_index") -> FAISS:
        """
        Load a vector store from the local file system.
        
        Parameters:
            store_path (str): Path where the FAISS index is stored.
        
        Returns:
            FAISS: Loaded FAISS vector store.
        """
        # embeddings = OllamaEmbeddings(model=self.model)
        embeddings=HuggingFaceEmbeddings(model_name=self.model)
        vector_store = FAISS.load_local(store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        logging.info(f"Vector store loaded from '{store_path}'.")
        return vector_store

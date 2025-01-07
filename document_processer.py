import os
import glob
from typing import List
from multiprocessing import Pool
import logging
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DocumentProcessor:
    """
    A pipeline for ingesting and processing text documents.

    Parameters:
        source_directory (str): Directory containing text files to process.
    """

    def __init__(self, source_directory: str):
        self.source_directory = source_directory

        # Validate directories
        if not os.path.exists(self.source_directory):
            raise ValueError(
                f"Source directory '{self.source_directory}' does not exist.")

    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single text file and return its content as a list of Document objects.

        Parameters:
            file_path (str): Path to the text file.

        Returns:
            List[Document]: List of loaded documents.
        """
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            return []  # Return an empty list for failed loads

    def load_documents(self) -> List[Document]:
        """
        Load all text files in the source directory recursively and process them in parallel.

        Returns:
            List[Document]: Combined list of all loaded documents.
        """
        # Get all text files recursively from the source directory
        all_files = glob.glob(os.path.join(
            self.source_directory, "**/*.txt"), recursive=True)
        if not all_files:
            logging.warning(
                f"No text files found in source directory '{self.source_directory}'.")
            return []

        results = []
        with Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=len(all_files), desc='Loading text documents', ncols=80) as pbar:
                for docs in pool.imap_unordered(self.load_single_document, all_files):
                    results.extend(docs)
                    pbar.update()

        return results

    def process_documents(self, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """
        Process and split documents into smaller chunks.

        Parameters:
            chunk_size (int): Maximum size (in tokens) of each chunk.
            chunk_overlap (int): Overlap (in tokens) between consecutive chunks.

        Returns:
            List[Document]: List of document chunks.
        """
        if chunk_size <= 0 or chunk_overlap < 0:
            raise ValueError(
                "chunk_size must be positive and chunk_overlap cannot be negative.")

        logging.info(f"Loading documents from '{self.source_directory}'")
        documents = self.load_documents()
        if not documents:
            logging.error("No text documents to process.")
            raise RuntimeError("No text documents to process.")

        logging.info(
            f"Loaded {len(documents)} text documents from '{self.source_directory}'")

        # text_splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
        # chunks = text_splitter.split_documents(documents)
        text_splitter=SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks=text_splitter.split_documents(documents)

        logging.info(
            f"Split into {len(chunks)} chunks of text (max. {chunk_size} tokens each).")
        return chunks

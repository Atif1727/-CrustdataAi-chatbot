import streamlit as st
from flask import Flask, request
from document_processer import DocumentProcessor
from vector_store_manager import VectorStoreManager
from reteriver import QaChainBuilder
import os


class CrustData:
    def __init__(self, source_directory):
        super(CrustData, self).__init__()
        if "messages" not in st.session_state:
            # Initialize chat messages with an opening message from the assistant
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I'm your CrustData API assistant. I can help you with:\n"
                        "- API endpoint documentation\n"
                        "- Example API requests\n"
                        "- Common issues and solutions\n"
                        "- Best practices\n"
                        "\nWhat would you like to know?"
                    )
                }
            ]

        self.vector_store_manager = VectorStoreManager()
        self.document_processor = DocumentProcessor(source_directory=source_directory)
        self.qa_chain_builder = QaChainBuilder()

    def process_document_store_vector_db(self, chunk_size=1000, chunk_overlap=50, store_path="faiss_index"):
        text_chunks = self.document_processor.process_documents(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store_manager.create_vector_store(
            text_chunks, store_path=store_path)

    def get_response(self, input_query, groq_api_key):
        docs = self.vector_store_manager.load_vector_store().similarity_search(input_query)
        chain = self.qa_chain_builder.build_chain(groq_api_key)

        # Construct context with previous messages
        context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
        )

        # Combine context with the current user question
        query_with_context = f"{context}\n\nUser: {input_query}"

        # Get response from the chain
        response = chain.invoke(
            {"input_documents": docs, "question": query_with_context}, return_only_outputs=True)

        return response

    def display_chat(self):
        """Display all chat messages in the session."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        """Handle user input from the chat interface."""
        if input_query := st.chat_input("CrustData AI Chatbot"):
            # Add user input to the chat history
            st.session_state.messages.append(
                {"role": "user", "content": input_query})
            with st.chat_message("user"):
                st.markdown(input_query)

            # Get the Groq API key from the environment variable
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("Groq API key is not set. Please set it in the sidebar.")
                return

            # Get the response from the model
            response = self.get_response(input_query, groq_api_key)
            assistant_response = response['output_text']

            # Display and store the assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response})

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(page_title="CrustData Chatbot", page_icon=":robot:")
        st.markdown(
            "<h1 style='color: #3498db; text-align: center;'>CrustData AI Chatbot</h1>", unsafe_allow_html=True)

        with st.sidebar:
            st.title("API Key Setup:")

            # Input fields for API keys
            groq_key = st.text_input(
                "Enter your Groq API Key:", type="password")

            if st.button("Submit API Keys"):
                if groq_key:
                    # Set the keys in the environment variables
                    os.environ["GROQ_API_KEY"] = groq_key
                    st.success("API keys have been set successfully!")
                else:
                    st.error("Please provide both OpenAI and Groq API keys.")

        # Display chat messages and handle user input
        self.display_chat()
        self.handle_user_input()


if __name__ == "__main__":
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'raw_txt')
    app = CrustData(source_directory)
    app.run()

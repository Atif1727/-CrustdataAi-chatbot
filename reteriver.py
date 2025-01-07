from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


class QaChainBuilder:

    def __init__(self, model="mixtral-8x7b-32768"):
        self.model = model

    def build_chain(self, groq_api_key):
        prompt_template = """
        Answer the question in as detailed manner as possible from the provided context. 
        If the answer is not in the provided context, just say, 
        "Answer is not available in the context." Do not provide a wrong answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        # Initialize the language model
        llm_model = ChatGroq(
            model=self.model, temperature=0.3, groq_api_key=groq_api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=[
                                "context", "question"])
        chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=prompt)

        return chain

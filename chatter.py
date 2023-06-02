import os
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import pinecone
from dotenv import load_dotenv

load_dotenv()

import openai

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
DIRECTORY_PATH = '/content/sample_data/test'
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
PROJECT_NAME = os.getenv('PROJECT_NAME')

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings.embed_query, "text")

prompt_template0 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer in 3-4 sentences

{context}

Question: {question}
Answer:"""
PROMPT0 = PromptTemplate(
    template=prompt_template0, input_variables=["context", "question"]
)

prompt_template1 = """You are friendly helpful bot , answer user travel related queries from you knowledgebase, also use
the following pieces of context below to arrive at the best answer. 

{context}

{chat_history}
Question: {question}
Answer:"""
PROMPT1 = PromptTemplate(
    template=prompt_template1, input_variables=["context", "chat_history", "question"]
)

docsearch = Pinecone.from_existing_index(PINECONE_INDEX, embeddings, namespace=PROJECT_NAME)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT1,
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question"),
    }
)


def get_response(user_message):
    global qa
    response = qa({"query": user_message})
    print(response)
    return response['result']


def reset():
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": PROMPT1,
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question"),
        }
    )

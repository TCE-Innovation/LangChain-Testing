# from dotenv import load_dotenv
# import os
# from langchain.llms import OpenAI
# from langchain import PromptTemplate, LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain.document_loaders import PDFMinerLoader, WebBaseLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# load_dotenv()
# openai_key = os.getenv("OPENAI_KEY")
# llm = ChatOpenAI(temperature = 0.5, openai_api_key=openai_key)
# # llm = OpenAI(openai_api_key=openai_key)

# # Load the PDF
# loader = PDFMinerLoader(file_path=r"C:\Users\roneill\OneDrive - Iovino Enterprises, LLC\Documents 1\
#                         Code\Git Files\Langchain Testing\C32520_PRDC01 Scope of Work 2023.06.22.pdf")
# pdf = loader.load()

# index = VectorstoreIndexCreator().from_loaders([pdf])

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
# all_splits = text_splitter.split_documents(pdf)

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# question = "What information is there on fuses?"
# docs = vectorstore.similarity_search(question)
# print(f"The size of docs is {len(docs)}")

# qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
# qa_chain({"query": question})

# # Create the LLMChain
# llm_chain = LLMChain(
#     llm,
#     document_loaders=[pdf],
# )

# # Create the initial messages
# messages = [
#     SystemMessage(
#         content=(
#                 """ You are a bot that has access to construction related documents.
#                 Based on the human's prompt, you will find all relevant information related
#                 to the human's prompt and provide citations as to where you found the information.
#                 Use only information from the supplied pdf.
#                 The citations must state the page number. """
#             )
#     ),
#     HumanMessage(
#         content="Find me all information in the pdf related to fuses."
#     ),
# ]

# # Generate a response
# response = llm_chain.generate_response(messages)

# # Print the response
# print(response)

# import os
# from dotenv import load_dotenv
# import openai
# import fitz  # PyMuPDF
# from langchain.document_loaders import TextDocument
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.retrievers import VectorstoreRetriever
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# load_dotenv()

# # Set your OpenAI API key
# openai_key = os.getenv("OPENAI_KEY")

# # Load and extract text from the PDF file
# pdf_file_path = "C32520_PRDC01 Scope of Work 2023.06.22.pdf"

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# pdf_text = extract_text_from_pdf(pdf_file_path)

# # Create LangChain Document
# pdf_document = TextDocument(pdf_text)

# # Split the document
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# pdf_splits = text_splitter.split_documents(pdf_document)

# # Store the splits in a vector store
# vectorstore = Chroma.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())

# # Define the question
# question = "What is discussed in the section of the PDF about Environmental Scope of Work?"

# # Retrieve relevant splits
# retriever = VectorstoreRetriever(vectorstore)
# relevant_splits = retriever.get_relevant_documents(question)

# # Create a RetrievalQA chain
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# # Perform question-answering
# answer = qa_chain({"query": question})

# # Print the answer
# print("Question:", question)
# print("Answer:", answer["result"])

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessage

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from PyPDF2 import PdfReader

from dotenv import load_dotenv
import os



# Load the API key stored in .env file
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai_api_key

# Load pdf stored in file explorer
loader = PdfReader(r"C:\Users\roneill\OneDrive - Iovino Enterprises, LLC\Documents 1\Code\Git Files\Langchain Testing\C32520_PRDC01 Scope of Work 2023.06.22.pdf")
# pages = loader.load_and_split()

from typing_extensions import Concatenate

# read text from pdf
raw_text = ''
for i, page in enumerate(loader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split the text using Character Text Split
# Doing it in a way to regulate token size
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)

# Download embeddings from OpenAi
embeddings = OpenAIEmbeddings(openai_api_key=key)

texts = text_splitter.split_text(raw_text)

document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(openai_api_key=key), chain_type ="stuff")

query = """
Do I need flagging to access the Reserve EDR?
Cite where you got your information.
State the section as well as the page number.
Quote the text you referred to.
"""

docs = document_search.similarity_search(query)
print(chain.run(input_documents=docs, question=query))



# template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                 """You are a helpful assistant that summarizes the 
#                 user's input text about Platform Screen Doors (PSDs).
#                 """
#             )
#         ),
#         HumanMessage(
#             content=(
#                 """
#                 Write a separate paragraph about all content related to switchboards in the input text"""
#             )
#         ),
#     ]
# )





# llm = ChatOpenAI(openai_api_key=openai_key)
# print(llm(template.format_messages(text=pages[1].page_content)).content)
# print()


# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import SystemMessage, HumanMessage

# from langchain.chat_models import ChatOpenAI
# from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI

# from PyPDF2 import PdfReader

# from dotenv import load_dotenv
# import os


# # Load the API key stored in .env file
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")
# # os.environ["OPENAI_API_KEY"] = openai_api_key

# # Load the API key stored in the .env file
# load_dotenv()

# # Set the OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Store the OpenAI API key in the environment
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # Load pdf stored in file explorer
# loader = PdfReader("C32520_PRDC01 Scope of Work 2023.06.22.pdf")

# # read text from pdf
# raw_text = ''
# for i, page in enumerate(loader.pages):
#     content = page.extract_text()
#     if content:
#         raw_text += content

# # Split the text using RecursiveCharacterTextSplitter
# # Doing it in a way to regulate token size
# text_splitter = RecursiveCharacterTextSplitter(
#     separator="\n",
#     chunk_size = 800,
#     chunk_overlap = 200,
#     length_function = len,
# )

# # Download embeddings from OpenAi
# embeddings = OpenAIEmbeddings(openai_api_key)

# texts = text_splitter.split_text(raw_text)

# # Create the FAISS vector store
# document_search = FAISS.from_texts_and_embeddings(texts, embeddings)

# chain = load_qa_chain(OpenAI(), chain_type ="stuff")

# query = "What is this document about?"
# docs = document_search.similarity_search(query)
# chain.run(input_documents=docs, question=query)

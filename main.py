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


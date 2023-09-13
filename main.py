from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.document_loaders import PyPDFLoader

template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """You are a helpful assistant that summarizes the 
                user's input text about Platform Screen Doors (PSDs).
                Write a separate paragraph about all content related to fuses in the input text"""
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

from langchain.chat_models import ChatOpenAI

loader = PyPDFLoader("C32520_PRDC01 Scope of Work 2023.06.22.pdf")
pages = loader.load_and_split()

llm = ChatOpenAI(openai_api_key="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")
print(llm(template.format_messages(text=pages[1].page_content)).content)
print()

# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
# from langchain.document_loaders import PyPDFLoader

# template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                 "You are a helpful assistant that summarizes the user's input text."
#             )
#         ),
#         HumanMessagePromptTemplate.from_template("{text}"),
#     ]
# )

# from langchain.chat_models import ChatOpenAI

# loader = PyPDFLoader("C32520_PRDC01 Scope of Work 2023.06.22.pdf")
# pages = loader.load_and_split()

# def get_user_input():
#     """Get user input from the keyboard."""
#     user_input = input("Enter the text you want to summarize: ")
#     return user_input

# llm = ChatOpenAI(openai_api_key="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")
# print(llm(template.format_messages(text=get_user_input())).content)
# print()

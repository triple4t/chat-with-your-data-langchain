from langchain_community.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv, find_dotenv
_= load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.7)


pdf_file_path = "Harry Potter and the Sorcerers Stone.pdf" # here goes your file path 
pdf_loader = PyPDFLoader(pdf_file_path)

docs = pdf_loader.load_and_split()

question = "summarize the chapter 1 from the book in 100 words."


prompt = """
Given the following extracted parts of a long document and a question.\n
If you don't knowthe answer, just say that you don't know. Don't try to make up an answer.\
n\n\nQUESTION: {question}\
n=========\nContent:
"""



chain = load_summarize_chain(llm, chain_type="map_reduce")
print(chain.run(docs))


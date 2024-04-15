from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GooglePalm
from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader("data.txt")
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(data)

embedder = GooglePalmEmbeddings(google_api_key="AIzaSyD8YpfmSZwp4-3kRxnILpLuflQ0xk-gaj0")

vectordb = Chroma.from_documents(chunks, embedder)

llm = GooglePalm(temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
)

query = "Какой главный вывод можно сделать из текста?"

result = qa_chain({"query": query})

print(result['result'])
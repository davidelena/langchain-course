import os

from dotenv import load_dotenv
from langchain_community.chat_loaders import WhatsAppChatLoader
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    # loader = TextLoader("/Users/dailiwei/Desktop/langchain-course/mediumblog.txt")
    loader = PDFPlumberLoader("/Users/dailiwei/Desktop/langchain-course/weicheng.pdf")

    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072,
                                  openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"ingesting batch {i // batch_size + 1} ({len(batch)} chunks)...")
        PineconeVectorStore.from_documents(
            batch, embeddings, index_name=os.environ["INDEX_NAME"]
        )
    print("finish")

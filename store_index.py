from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone


load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


extracted_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
embeddings= download_hugging_face_embeddings()



pinecone_api_key = PINECONE_API_KEY
pc= Pinecone(api_key=pinecone_api_key)

index_name= "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec= ServerlessSpec(cloud="aws",region="us-east-1")
    )

index= pc.Index(index_name)

stats = index.describe_index_stats()

# upload only if index is empty 
if stats.total_vector_count == 0:
    print("Index is empty, uploading documents...")

    docsearch= PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
else:
    print("Doc exists, skipping upload")
    docsearch= PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
        )
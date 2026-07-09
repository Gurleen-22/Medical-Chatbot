from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document



# Extract text from PDF files
def load_pdf_files(data):
    loader =DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

 

#to filter out the actual data from the pdf files

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    to filter out - given a list of documnent objects, return a new list of document objects 
    that contain only the source in metadata and original page content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src= doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content =doc.page_content,
                metadata={"Source":src}
            )
        )
    return minimal_docs


# split the documents into smaller chunks
#Chunking 
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts_chunk= text_splitter.split_documents(extracted_data)
    return texts_chunk



def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

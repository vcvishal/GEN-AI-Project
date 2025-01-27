
#importing Libraries
import os
os.environ['OPENAI_API_KEY'] = use you key
HUGGINGFACEHUB_API_TOKEN = 'hf_KAwpdpPtbejjZIONUPWqaV'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter
 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
import pandas as pd
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer

# # HuggingFace model setup
repo_id = "google/flan-t5-xxl"
#hf_model = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 64})
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                     streaming=True)
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Welcome to Question Answer RAG Chatbot 🤖")


class HuggingFaceEmbeddingsWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, query):
        """Embed a single query."""
        return self.model.encode([query], convert_to_tensor=False).tolist()[0]

# Initialize the embedding wrapper

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(embedding_model_name)
huggingface_embeddings = HuggingFaceEmbeddingsWrapper(sentence_model)

# Folder containing PDF files
folder_path = "dataset"

# Load documents from PDFs
def load_documents(folder_path):
    doc_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            doc_list.extend(docs)
    return doc_list
doc_list=load_documents(folder_path)

#retrivers 
def create_openai_retriever(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model, collection_name="openai_embeddings")
    return vectordb.as_retriever()


def create_huggingface_retriever(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    vectordb = Chroma.from_documents(doc_chunks, huggingface_embeddings, collection_name="hf_embeddings")
    return vectordb.as_retriever()

retriever_hg=create_huggingface_retriever(doc_list)
retriever_oi=create_openai_retriever(doc_list)
# #print(doc_list)

### Similarity or Ranking based Retrieval

def create_openai_retriever_sim(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model, collection_name="openai_embeddings")
    return vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})


def create_huggingface_retriever_sim(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    vectordb = Chroma.from_documents(doc_chunks, huggingface_embeddings, collection_name="hf_embeddings")
    return vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
retriever_hg=create_openai_retriever_sim(doc_list)
retriever_oi=create_huggingface_retriever_sim(doc_list)

##  Multiquery
from langchain.retrievers.multi_query import MultiQueryRetriever

def create_huggingface_retriever_mq(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    vectordb = Chroma.from_documents(doc_chunks, huggingface_embeddings, collection_name="hf_embeddings")
    sm=vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    return MultiQueryRetriever.from_llm(
    retriever=sm, llm=chatgpt
)

def create_openai_retriever_mq(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model, collection_name="openai_embeddings")
    sm=vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    return MultiQueryRetriever.from_llm(
    retriever=sm, llm=chatgpt
)

retriever_oi=create_openai_retriever_mq(doc_list)
retriever_hg=create_huggingface_retriever_mq(doc_list)



##Reranker


def create_openai_retriever_reranker(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(doc_list)
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model, collection_name="openai_embeddings")
    sm=vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    _filter = LLMChainFilter.from_llm(llm=chatgpt)
    # Retriever 2 - retrieves the documents similar to query and then applies the filter
    compressor_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=sm
    )
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
    reranker_compressor = CrossEncoderReranker(model=reranker, top_n=3)
    return ContextualCompressionRetriever(
    base_compressor=reranker_compressor, base_retriever=compressor_retriever
)
retriever_hg=create_openai_retriever_reranker(doc_list)

if not doc_list:
    st.error("No PDF documents found in the 'dataset' folder.")
else:
    st.success(f"Loaded {len(doc_list)} documents.")

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)




# Load a connection to ChatGPT LLM
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                     streaming=True)

# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

# Create a QA RAG System Chain
qa_rag_chain_oi = (
  {
    "context": itemgetter("question") # based on the user question get context docs
      |
    retriever_oi
      |
    format_docs,
    "question": itemgetter("question") # user question
  }
    |
  qa_prompt # prompt with above user question and context
    |
  chatgpt # above prompt is sent to the LLM for response
)

qa_rag_chain_hg = (
  {
    "context": itemgetter("question") # based on the user question get context docs
      |
    retriever_hg
      |
    format_docs,
    "question": itemgetter("question") # user question
  }
    |
  qa_prompt # prompt with above user question and context
    |
  chatgpt # above prompt is sent to the LLM for response
)

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
  streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
  st.chat_message(msg.type).write(msg.content)


class PostMessageHandler(BaseCallbackHandler):
  def __init__(self, msg: st.write):
    BaseCallbackHandler.__init__(self)
    self.msg = msg
    self.sources = []

  def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
    source_ids = []
    for d in documents: # retrieved documents from retriever based on user query
      metadata = {
        "source": d.metadata["source"],
        "page": d.metadata["page"],
        "content": d.page_content[:200]
      }
      idx = (metadata["source"], metadata["page"])
      if idx not in source_ids: # store unique source documents
        source_ids.append(idx)
        self.sources.append(metadata)

  def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
    if len(self.sources):
      st.markdown("__Sources:__ "+"\n")
      st.dataframe(data=pd.DataFrame(self.sources[:3]),
                    width=1000) # Top 3 sources





if user_prompt := st.chat_input():
    # Display the user's input
    st.chat_message("human").write(user_prompt)

    # OpenAI Response Handling
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_container = st.empty()  # Dynamic container for sources
        pm_handler = PostMessageHandler(sources_container)

        config = {
            "configurable": {"session_id": "any"},
            "callbacks": [stream_handler, pm_handler]
        }

        try:
            response_oi = qa_rag_chain_oi.invoke({
                "question": user_prompt,
                "context": lambda question: format_docs(retriever.get_relevant_documents(question)),
            }, config)


        except Exception as e:
            st.error(f"An error occurred with OpenAI chain: {e}")

    # HuggingFace Response Handling
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_container = st.empty()
        pm_handler = PostMessageHandler(sources_container)

        config = {
            "configurable": {"session_id": "any"},
            "callbacks": [stream_handler, pm_handler]
        }

        try:
            response_hg = qa_rag_chain_hg.invoke({"question": user_prompt}, config)



        except Exception as e:
            st.error(f"An error occurred with HuggingFace chain: {e}")

       

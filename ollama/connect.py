import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
PDF_PATH = Path(__file__).with_name("NIPS-2017-attention-is-all-you-need-Paper.pdf")

loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
test_vector = emb.embed_query("Transformer architecture")
print(f"Dimensiones del vector: {len(test_vector)}")
print(f"Primeros 5 números: {test_vector[:5]}")

vs = Chroma.from_documents(chunks, emb, persist_directory=CHROMA_PERSIST_DIR)
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4})

ctx = retriever.invoke("Explain the Scaled Dot-Product Attention mechanism described in the paper")
llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

context_parts = []
for i, d in enumerate(ctx, start=1):
    page = d.metadata.get("page", "N/A")
    text = d.page_content
    print(f"[ctx {i}/{len(ctx)}] page={page} chars={len(text)}")
    context_parts.append(text)

prompt = "Resume esto:\n\n" + "\n\n".join(context_parts)
print(f"[prompt] chars={len(prompt)}")
resp = llm.invoke(prompt)
print(resp.content)

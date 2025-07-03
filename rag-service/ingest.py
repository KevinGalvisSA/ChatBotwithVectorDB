import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant


# 1. Cargar variables de entorno
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# 2. Cargar documento PDF
loader = PyPDFLoader("data/constitucion_colombia.pdf")
documents = loader.load()

# 3. Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 4. Embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Cliente Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 6. Crear colección si no existe
if QDRANT_COLLECTION not in [col.name for col in client.get_collections().collections]:
    print("📦 Creando colección en Qdrant...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
else:
    print("✅ Colección ya existe en Qdrant")

# 7. Guardar documentos en Qdrant
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=QDRANT_COLLECTION,
    client=client
)

print(f"✅ {len(chunks)} documentos cargados y embebidos en Qdrant con éxito.")

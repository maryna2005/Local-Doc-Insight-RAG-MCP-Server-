import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_DEVICE"] = "cpu"
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pypdf import PdfReader

# 1. Initialize MCP and Embedding Model
mcp = FastMCP("Advanced-RAG-Assistant")
# This model converts text into numerical vectors (Embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Storage for our text chunks
documents = []
index = None

@mcp.tool()
def load_documents_from_folder(folder_path: str) -> str:
    """
    Reads PDF/TXT files and builds a searchable vector index.
    Strategic Value: Cost Efficiency - local processing without retraining.
    """
    global index, documents
    documents = []
    
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    for file in os.listdir(folder_path):
        text = ""
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + " "
        elif file.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if text:
            # Split text into chunks of 500 characters for better search precision
            chunks = [text[i:i+500] for i in range(0, len(text), 400)]
            documents.extend(chunks)

    if not documents:
        return "No text found in the documents."

    # Create Vector Index using FAISS (Facebook AI Similarity Search)
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return f"Successfully indexed {len(documents)} text chunks from {folder_path}."

@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """
    Retrieves relevant info from the indexed data to answer questions.
    Strategic Value: Accuracy & Trust - grounds AI in your real data.
    """
    if index is None or not documents:
        return "Knowledge base is empty. Please load documents first."

    # Search for the top 3 most relevant context chunks
    question_enc = model.encode([question])
    distances, indices = index.search(np.array(question_enc).astype('float32'), k=3)
    
    context = "\n---\n".join([documents[i] for i in indices[0]])
    return f"Relevant information found in your documents:\n\n{context}"

if __name__ == "__main__":
    mcp.run()
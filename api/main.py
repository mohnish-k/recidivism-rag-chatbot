import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directory to path to import rag_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG system components
from rag_system.vector_store import FAISSVectorStore
from rag_system.openai_client import OpenAIClient
from rag_system.retriever import Retriever
from rag_system.response_generator import ResponseGenerator

# Load environment variables
load_dotenv()
print(f"Current working directory: {os.getcwd()}")
print(f"Environment variables loaded: VECTOR_DB_PATH={os.getenv('VECTOR_DB_PATH')}")

app = FastAPI(title="Recidivism Research RAG API")

# Configure CORS to allow requests from your Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    
# Singleton pattern for RAG components to avoid reinitializing for each request
rag_components = {}

def get_rag_system():
    """Initialize and return RAG system components (singleton pattern)"""
    global rag_components
    
    if not rag_components:
        try:
            print("Initializing RAG system components...")
            # Initialize components
            vector_store = FAISSVectorStore()
            print(f"FAISS index loaded with {vector_store.index.ntotal} vectors")
            
            openai_client = OpenAIClient()
            print("OpenAI client initialized")
            
            retriever = Retriever(vector_store)
            print("Retriever initialized")
            
            response_generator = ResponseGenerator(openai_client)
            print("Response generator initialized")
            
            rag_components = {
                "vector_store": vector_store,
                "openai_client": openai_client,
                "retriever": retriever,
                "response_generator": response_generator
            }
            print("RAG system components initialized successfully")
            
        except Exception as e:
            import traceback
            print(f"Error initializing RAG system: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {str(e)}")
    
    return rag_components

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(f"Received chat request: {request.query}")
        # Get RAG components
        rag_system = get_rag_system()
        retriever = rag_system["retriever"]
        response_generator = rag_system["response_generator"]
        
        # Retrieve relevant context
        print(f"Retrieving context for query: {request.query}")
        context_items = retriever.retrieve_context(request.query)
        print(f"Retrieved {len(context_items)} context items")
        
        # If no context items, return a specific message
        if not context_items or len(context_items) == 0:
            print("No context items found - returning default message")
            return ChatResponse(
                answer="I couldn't find any relevant information in my knowledge base to answer your question. This could be due to a data retrieval issue or the information may not be present in my research papers.",
                sources=[]
            )
        
        # Generate a response using the retrieved context
        print("Generating response...")
        response = response_generator.generate_response(
            query=request.query,
            context_items=context_items,
            conversation_history=request.conversation_history
        )
        
        # Format sources for citation
        sources = [
            {
                "document_id": str(item.get("document_id")),
                "filename": item.get("filename", "Unknown document"),
                "relevance_score": float(item.get("score", 0.0))
            }
            for item in context_items
        ]
        
        print(f"Response generated successfully with {len(sources)} sources")
        return ChatResponse(answer=response, sources=sources)
    
    except Exception as e:
        import traceback
        print(f"Error processing chat request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
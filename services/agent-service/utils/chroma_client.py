import json
import logging
import chromadb
from config.settings import settings
from utils.aws_bedrock import generate_text_embedding

logger = logging.getLogger(__name__)

# Initialize ChromaDB HTTP Client
try:
    # Since ChromaDB runs as a service in docker-compose, its hostname is "chromadb"
    chroma_client = chromadb.HttpClient(
        host=settings.REDIS_HOST.replace("redis", "chromadb"), # fallback to chromadb host on docker net
        port=8000
    )
    logger.info("ChromaDB Client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB Client: {e}")
    chroma_client = None

def _get_user_collection(user_id: str):
    if chroma_client is None:
        raise Exception("ChromaDB client is not initialized")
    # Collections in Chroma must be between 3 and 63 chars, start/end with alphanumeric, contain no double dots
    collection_name = f"user_{user_id.replace('-', '_')}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def index_image_vector(image_id: str, user_id: str, scene_description: str, detected_faces: list[str]) -> bool:
    try:
        collection = _get_user_collection(user_id)
        
        # Generate text embedding
        vector = generate_text_embedding(scene_description)
        
        # Meta dictionary
        meta = {
            "user_id": user_id,
            "detected_faces": json.dumps(detected_faces)
        }
        
        # Upsert into Chroma
        collection.upsert(
            ids=[image_id],
            embeddings=[vector],
            documents=[scene_description],
            metadatas=[meta]
        )
        logger.info(f"Indexed image vector {image_id} in ChromaDB successfully")
        return True
    except Exception as e:
        logger.error(f"Error indexing vector in ChromaDB: {e}")
        return False

def deindex_image_vector(image_id: str, user_id: str) -> bool:
    try:
        collection = _get_user_collection(user_id)
        collection.delete(ids=[image_id])
        logger.info(f"Deleted vector {image_id} from ChromaDB")
        return True
    except Exception as e:
        logger.error(f"Error deindexing vector from ChromaDB: {e}")
        return False

def search_image_vectors(user_id: str, query_text: str, limit: int = 15) -> list[dict]:
    if chroma_client is None:
        logger.warning("ChromaDB client is not available. Returning empty search results.")
        return []
        
    try:
        collection = _get_user_collection(user_id)
        
        # Generate search query embedding
        query_vector = generate_text_embedding(query_text)
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit
        )
        
        parsed_results = []
        if results and results["ids"]:
            # Chroma returns nested arrays for ids, distances, documents, metadatas
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            
            for img_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
                # Convert distance to similarity score
                score = 1.0 - dist
                
                faces_str = meta.get("detected_faces", "[]")
                try:
                    detected_faces = json.loads(faces_str)
                except Exception:
                    detected_faces = []
                    
                parsed_results.append({
                    "image_id": img_id,
                    "score": float(score),
                    "document": doc,
                    "detected_faces": detected_faces
                })
                
        return parsed_results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []

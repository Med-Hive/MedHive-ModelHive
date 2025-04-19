"""
Database service for AstraDB vector store operations
"""
import logging
from fastapi import HTTPException
from langchain_astradb import AstraDBVectorStore
from langchain_core.embeddings import FakeEmbeddings
from app.config import *
from app.services.disease import process_csv_data
from astrapy import DataAPIClient, Database
logger = logging.getLogger(__name__)


def connect_to_database() -> Database:
    """
    Connects to a DataStax Astra database.
    This function retrieves the database endpoint and application token from the
    environment variables `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN`.

    Returns:
        Database: An instance of the connected database.

    Raises:
        RuntimeError: If the environment variables `ASTRA_DB_API_ENDPOINT` or
        `ASTRA_DB_APPLICATION_TOKEN` are not defined.
    """

    endpoint = f"{ASTRA_DB_API_ENDPOINT}"
    print(endpoint)
    token = F"{ASTRA_DB_APPLICATION_TOKEN}"

    if not token or not endpoint:
        raise RuntimeError(
            "Environment variables ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN must be defined"
        )

    # Create an instance of the `DataAPIClient` class
    client = DataAPIClient()

    # Get the database specified by your endpoint and provide the token.
    database = client.get_database(endpoint, token=token)

    print(f"Connected to database {database.info().name}")

    return database

def get_astra_db():
    db=connect_to_database()

    """Initialize and return AstraDB vector store connection"""
    embedding_model = FakeEmbeddings(size=1536)  # Using fake embeddings for now
    
    try:
        vector_store = AstraDBVectorStore(
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            api_endpoint=f"{ASTRA_DB_API_ENDPOINT}",
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to AstraDB: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

async def initialize_database():
    """Check if collection exists and upload data if needed"""
    vector_store = get_astra_db()
    
    # Check if collection exists and has data
    try:
        # Simple query to check if collection has data
        results = vector_store.similarity_search("test", k=1)
        if results:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists with data")
            return
    except Exception:
        # Collection may be empty or doesn't exist
        pass
    
    # Process CSV and upload to AstraDB
    logger.info(f"Initializing collection '{COLLECTION_NAME}' with data")
    documents = process_csv_data()
    
    if documents:
        try:
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to AstraDB collection")
        except Exception as e:
            logger.error(f"Failed to add documents to AstraDB: {e}")
            raise HTTPException(status_code=500, detail=f"Database upload error: {str(e)}")
    else:
        logger.warning("No documents to add to the database")

def search_diseases_by_symptoms(symptoms, vector_store):
    """Search for diseases based on symptoms"""
    if not symptoms:
        return "No specific symptoms detected in your message."
    
    # Create a query from the symptoms
    query = f"Symptoms: {', '.join(symptoms)}"
    
    # Search for similar documents
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No exact matches found in our database for these symptoms."
    
    # Format the results
    formatted_results = []
    for doc in results:
        disease = doc.metadata.get("disease", "Unknown condition")
        matched_symptoms = doc.metadata.get("symptoms", [])
        common_symptoms = set(symptoms) & set(matched_symptoms)
        
        if common_symptoms:
            match_info = {
                "disease": disease,
                "matched_symptoms": list(common_symptoms),
                "all_disease_symptoms": matched_symptoms
            }
            formatted_results.append(match_info)
    
    if not formatted_results:
        return "No relevant matches found for your symptoms."
    
    # Create a readable response
    response = "Based on the symptoms you mentioned, here are potential matches:\n\n"
    for i, result in enumerate(formatted_results, 1):
        response += f"{i}. {result['disease']}\n"
        response += f"   Matching symptoms: {', '.join(result['matched_symptoms'])}\n"
        response += f"   Other symptoms of this condition: {', '.join([s for s in result['all_disease_symptoms'] if s not in result['matched_symptoms']])}\n\n"
    
    return response

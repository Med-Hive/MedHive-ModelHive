"""
Disease data processing services
"""
import pandas as pd
import logging
from fastapi import HTTPException
from langchain_core.documents import Document
from app.config import DATA_CSV_PATH

logger = logging.getLogger(__name__)

def process_csv_data(file_path=DATA_CSV_PATH):
    """Process the CSV data and create document list"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Process data to create document list
        documents = []
        diseases = df.iloc[::2].reset_index(drop=True)  # Every other row contains disease names
        
        for i in range(len(diseases)):
            disease_name = df.columns[0]  # First column has disease name
            symptoms_row = df.iloc[i*2]  # Disease name row
            values_row = df.iloc[i*2+1]  # Values row (0/1)
            
            # Get the actual disease name from the first cell
            disease = symptoms_row[0]
            
            # Create symptom list where value is 1
            symptoms = []
            for j in range(1, len(symptoms_row)):
                if values_row[j] == 1:
                    symptoms.append(df.columns[j])
            
            # Create document
            if symptoms:
                content = f"Disease: {disease}\nSymptoms: {', '.join(symptoms)}"
                metadata = {
                    "disease": disease,
                    "symptoms": symptoms
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV data: {str(e)}")

def get_all_symptoms(file_path=DATA_CSV_PATH):
    """Extract all possible symptoms from the CSV file"""
    try:
        with open(file_path, "r") as f:
            header = f.readline().strip().split(",")
            all_symptoms = header[1:]  # Skip the first column which is the disease name
        return all_symptoms
    except Exception as e:
        logger.error(f"Error getting symptoms list: {e}")
        return []

def extract_symptoms(user_input, all_symptoms):
    """Extract symptoms from user input"""
    # Convert user input to lowercase for case-insensitive matching
    user_input_lower = user_input.lower()
    
    # Find all symptoms mentioned in the user input
    found_symptoms = []
    for symptom in all_symptoms:
        if symptom.lower() in user_input_lower:
            found_symptoms.append(symptom)
    
    return found_symptoms

import groq
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.models.responses import Disease, SymptomAnalysisResponse
from astrapy.db import AstraDB
import time
import uuid

settings = get_settings()

class ChatService:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self.model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        self.astra_db = AstraDB(
            token=settings.ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=settings.ASTRA_DB_API_ENDPOINT
        )
        self.collection = self.astra_db.collection("disease_vectors")
        self.current_model_version = str(uuid.uuid4())  # In production, get from model registry

    async def analyze_symptoms(self, symptoms: str, additional_context: str = None, user_id: Optional[str] = None) -> SymptomAnalysisResponse:
        start_time = time.time()
        try:
            # Generate embedding for the input symptoms
            symptoms_embedding = self.model.encode(symptoms).tolist()

            # Search for similar symptom patterns in AstraDB
            similar_cases = await self._search_similar_cases(symptoms_embedding)

            # Prepare context for LLM
            context = self._prepare_llm_context(similar_cases, symptoms, additional_context)

            # Get analysis from Groq
            completion = await self._get_groq_analysis(context)
            
            # Calculate latency for tracking
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Parse and validate the response
            try:
                analysis_dict = json.loads(completion)
                response = SymptomAnalysisResponse(**analysis_dict)
                
                # Track model tokens and performance
                tokens_used = len(str(context)) + len(completion)
                
                # Track prediction for monitoring
                self.model_monitor.track_prediction(
                    input_features=symptoms_embedding,
                    prediction=analysis_dict,
                    latency=latency,
                    ground_truth=None  # We don't have immediate feedback
                )
                

                
                # Track usage if user_id is provided
                if user_id:
                    request_data = {
                        "symptoms": symptoms,
                        "additional_context": additional_context
                    }
                
                
                return response
            except json.JSONDecodeError:
                return self._create_fallback_response(completion, similar_cases)
                
        except Exception as e:
            raise e


    async def _search_similar_cases(self, embedding: List[float]) -> List[Dict]:
        try:
            similar_cases = self.collection.find_many(
                filter={},
                sort={"$vector": embedding},
                limit=5
            )
            return similar_cases
        except Exception as e:
            raise e

    async def _get_groq_analysis(self, context: str) -> str:
        try:
            chat_completion = await self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical diagnostic AI assistant. Provide analyses in the requested JSON format. Be conservative with severity assessments and always recommend professional medical consultation."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model=settings.GROQ_MODEL_NAME,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise e

    def _prepare_llm_context(self, similar_cases: List[Dict], symptoms: str, additional_context: str = None) -> str:
        context = f"""As a medical diagnostic AI, analyze the following symptoms:
Symptoms: {symptoms}
"""
        if additional_context:
            context += f"Additional Context: {additional_context}\n"

        context += "\nSimilar cases from the database:\n"
        for case in similar_cases:
            context += f"- Disease: {case['disease']}\n  Symptoms: {case['symptoms_text']}\n"

        context += """\nProvide a detailed analysis in the following JSON format:
{
    "possible_diseases": [
        {
            "name": "disease name",
            "confidence": 0.0 to 1.0,
            "symptoms_matched": ["symptom1", "symptom2"],
            "recommendation": "specific recommendation"
        }
    ],
    "analysis_summary": "detailed analysis",
    "severity_level": "LOW/MEDIUM/HIGH",
    "seek_immediate_care": boolean
}"""

        return context

    def _create_fallback_response(self, completion: str, similar_cases: List[Dict]) -> SymptomAnalysisResponse:
        # Create a basic response when JSON parsing fails
        possible_diseases = [
            Disease(
                name=case["disease"],
                confidence=0.5,
                symptoms_matched=case["symptoms_text"].split(", "),
                recommendation="Please consult a healthcare professional for accurate diagnosis."
            )
            for case in similar_cases[:3]
        ]

        return SymptomAnalysisResponse(
            possible_diseases=possible_diseases,
            analysis_summary="Analysis could not be fully processed. Please consult a healthcare professional.",
            severity_level="MEDIUM",
            seek_immediate_care=True
        )
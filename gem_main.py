from fastapi import FastAPI, Request
from pydantic import BaseModel
# from qdrant_client import QdrantClient
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
import httpx
import re
import os
import subprocess

from dotenv import load_dotenv

load_dotenv(".env")

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Initialize FastAPI
app = FastAPI()

# Qdrant setup
# qdrant_client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY")
# )

# Embeddings and Vectorstore (assuming GATE-AraBERT model)
# embedding = HuggingFaceEmbeddings(model_name="aubmindlab/bert-base-arabertv2")
# embedding = HuggingFaceEmbeddings(
#     model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")

# Initialize Qdrant vectorstore
# qdrant_vectorstore = Qdrant(
#     client=qdrant_client,
#     collection_name="arabic_rag_collection",
#     embeddings=embedding
# )


# retriever = qdrant_vectorstore.as_retriever()

# RAG chain using Langchain
# rag_chain = RetrievalQA.from_chain_type(
#     llm=None,  # Gemini is used externally
#     retriever=qdrant_vectorstore.as_retriever(),
#     return_source_documents=False
# )

# Input model


class QueryInput(BaseModel):
    query: str
    patientContext: dict = {}

# Language detection (simple heuristic)


def detect_language(text: str) -> str:
    arabic_range = re.compile(r"[\u0600-\u06FF]")
    if arabic_range.search(text):
        return "ar"
    else:
        return "en"


SYSTEM_PROMPT = """You are Scanalyze AI, a bilingual medical assistant that speaks both Arabic and English. IMPORTANT: Always respond in the SAME LANGUAGE as the patient's question.

LANGUAGE RULES: reply in {language}

RESPONSE STYLE:
- Focus on the most essential information 
- Use clear, simple language
- Be direct and to the point
- USE bullet points or structured sections

IMPORTANT GUIDELINES:
1. Provide the most critical information
2. Use clear, simple language appropriate to the chosen language
3. Be professional and empathetic
4. PERSONALIZE responses based on patient information (gender, age, medical history)
5. Only mention gender-specific information that applies to THIS patient
6. Tailor advice to the patient's specific profile and circumstances

{context}

Please respond to the patient's question with a detailed answer. IMPORTANT: Personalize your response based on the patient's specific information (gender, age, medical history). Keep it concise and focus only on the most essential information.

Patient's Question: {question};
"""


async def call_gemini_api(prompt: str) -> str:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error: {response.status_code}, {response.text}"


def format_patient_context(context: dict) -> str:
    if not context:
        return ""
    formatted = "Patient Information:\n"
    for key, value in context.items():
        formatted += f"- {key.capitalize()}: {value}\n"
    return formatted.strip()


@app.get("/")
async def root():
    return {"message": "Welcome to the Scanalyze Chatbot."}


@app.post("/ask")
async def ask_question(data: QueryInput):
    query = data.query
    patient_context = data.patientContext
    lang = detect_language(query)

    # Remove RAG: do not retrieve context from Qdrant
    context_text = ""

    formatted_context = format_patient_context(patient_context)
    if lang == "ar":
        # context_block = f"{formatted_context}\n\nالسياق:\n{context_text}"

        prompt = SYSTEM_PROMPT.format(
            language=lang,
            context=formatted_context,
            question=query
        )
    else:
        context_block = formatted_context

        prompt = SYSTEM_PROMPT.format(
            language=lang,
            context=formatted_context,
            question=query
        )

    answer = await call_gemini_api(prompt)
    return {"language": "Arabic" if lang == "ar" else "English", "answer": answer}


# === RUN FASTAPI + STREAMLIT TOGETHER ===
def run_streamlit():
    subprocess.run(["streamlit", "run", "gemma_st.py"])


if __name__ == "__main__":

    import uvicorn
    import threading

    # Run Streamlit in a separate thread
    threading.Thread(target=run_streamlit, daemon=True).start()

    # Run FastAPI backend
    uvicorn.run(app, host="localhost", port=8080)

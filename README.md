## Scanalyze-Medical-MVP

- Scanalyze is an interactive medical analysis web application built with Streamlit and FastAPI. It integrates AI-powered diagnostic tools for analyzing medical images (Chest X-rays & Kidney CT scans) and a conversational medical chatbot to assist patients and practitioners.

### Technologies Used

- Streamlit – Frontend UI
- FastAPI – Backend APIs
- Uvicorn
- Requests – Communication between frontend and backends
- Threading – To run multiple backends simultaneously


### Clone the Repository
```
git clone https://github.com/khalednabawey/Scanalyze-MVP.git
cd Scanalyze-MVP
```

### Virtual Environment

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Enviroment Variables

```
GEMINI_API_KEY=GEMINI_API_KEY
```

### Running the App

```
streamlit run main_st.py
```

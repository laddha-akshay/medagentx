# MedAgentX  
Lightweight clinical assistant that retrieves relevant patient notes and generates structured outputs.

## Overview  
MedAgentX takes synthetic clinical notes, embeds them using MiniLM, indexes them with FAISS, retrieves the most relevant notes, and produces structured JSON answers using a schema first LLM prompt design. It is intended for safe demonstrations with synthetic data only.

## Features  
• Upload synthetic patient notes  
• Embedding generation with MiniLM  
• Fast similarity search using FAISS  
• Structured JSON responses  
• Pydantic model validation  
• Clean and simple UI  
• Works with or without OPENAI_API_KEY

## Architecture  
```
Notes → Embeddings → FAISS Index → Retrieval → JSON Guided LLM → Structured Output
```

### Detailed Diagram  
```
                 +----------------------+
                 |     User Interface   |
                 +-----------+----------+
                             |
                             v
                    Upload and Query
                             |
                 +-----------+-----------+
                 |         FastAPI        |
                 +-----------+-----------+
                             |
        +--------------------+----------------------+
        |                                           |
        v                                           v
   Ingestion                                Query Agent
(Load and clean notes)         (Retrieve notes and build JSON prompt)
        |                                           |
        v                                           v
Embedding Model (MiniLM)                 LLM or fallback summary
        |                                           |
        v                                           v
    FAISS Index                           Structured JSON output
```

## How to Run Locally  

### Create a virtual environment  
```bash
python -m venv venv
```

### Activate it  
```bash
source venv/bin/activate
```

### Install dependencies  
```bash
pip install -r requirements.txt
```

### Start the server  
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Open in browser  
```
http://localhost:8000
```

## Example Query  
Try something like:  
```
What are possible causes and next steps for this patient
```

## Example Output  
The system returns:  
• Relevant retrieved patient notes  
• JSON formatted: summary, findings, next steps, confidence score

## Developer Notes  
Set OPENAI_API_KEY for enhanced reasoning:  
```bash
export OPENAI_API_KEY="your_key_here"
```


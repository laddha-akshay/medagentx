# MedAgentX
Structured clinical assistant built on retrieval augmented generation.

## Overview
MedAgentX indexes synthetic patient notes, retrieves relevant context using FAISS, and returns structured JSON responses through a schema first LLM prompt design.

## How to Run
\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`
Open:
\`\`\`
http://localhost:8000
\`\`\`

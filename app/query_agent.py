import os
from typing import List
from pydantic import BaseModel
try:
    import openai
except Exception:
    openai = None
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if openai and OPENAI_KEY:
    openai.api_key = OPENAI_KEY
class ClinicalAnswer(BaseModel):
    summary: str
    findings: list
    suggested_next_steps: list
    confidence: float
def build_prompt(note_texts: List[str], question: str):
    context = '\n\n'.join([f'Note: {t[:1000]}' for t in note_texts])
    prompt = f"""You are a clinical assistant that summarizes and answers questions based on clinical notes.

Provide JSON with keys summary, findings, suggested_next_steps, confidence where confidence is 0 to 1.

Context:
{context}

Question: {question}

Return only valid JSON parsable into the schema summary string, findings list of short strings, suggested_next_steps list of short strings, confidence float."""
    return prompt
def answer_clinical(question: str, notes: List[dict]):
    texts = [n.get('text', '') for n in notes]
    prompt = build_prompt(texts, question)
    if openai and OPENAI_KEY:
        resp = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role':'user','content':prompt}],
            max_tokens=400
        )
        content = resp['choices'][0]['message']['content']
        import json
        try:
            obj = json.loads(content)
            return ClinicalAnswer(**obj)
        except Exception:
            return ClinicalAnswer(summary=content, findings=[], suggested_next_steps=[], confidence=0.0)
    else:
        summary = ' | '.join([t[:200] for t in texts])
        return ClinicalAnswer(summary=summary, findings=[], suggested_next_steps=[], confidence=0.0)

import google.generativeai as genai
from django.conf import settings

def generate_embedding(text):
    genai.configure(api_key=settings.GEMINI_API_KEY)
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

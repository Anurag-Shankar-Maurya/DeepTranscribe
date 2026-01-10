import google.genai as genai
from django.conf import settings

def generate_embedding(text):
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            task_type="retrieval_query"
        )
        return result.embedding
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

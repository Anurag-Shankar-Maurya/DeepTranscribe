from openai import OpenAI
from django.conf import settings

def generate_embedding(text):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

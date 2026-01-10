import json
from .models import Entity, KnowledgeTriple, HierarchicalSummary, ChatSession
from .retrieval_service import AdvancedRetrievalService

class MemoryConsolidator:
    """
    Runs in background after a chat session ends or transcript is uploaded.
    Populates the Knowledge Graph and Macro Summaries.
    """
    def __init__(self, client):
        self.client = client
        self.extraction_model = "gemma-2-27b-it"

    def process_session(self, session_obj):
        # 1. Get raw text
        messages = session_obj.messages.all()
        full_text = "\n".join([m.content for m in messages])

        # 2. Extract Entities & Relations (Graph)
        self._extract_graph_data(session_obj.user, full_text, session_obj)

        # 3. Create Hierarchical Summary (Macro)
        self._create_summary(session_obj.user, full_text)

    def _extract_graph_data(self, user, text, session_obj=None):
        prompt = f"""
        Extract knowledge triples from this text.
        Focus on persistent facts about the user, projects, people, and preferences.
        Format: JSON list of objects {{ "subject": "...", "predicate": "...", "object": "..." }}
        Text: {text[:10000]}
        """
        try:
            response = self.client.models.generate_content(
                model=self.extraction_model,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            triples = json.loads(response.text)

            for item in triples:
                sub, _ = Entity.objects.get_or_create(
                    name=item['subject'],
                    user=user,
                    defaults={'entity_type': 'UNKNOWN'}
                )
                obj, _ = Entity.objects.get_or_create(
                    name=item['object'],
                    user=user,
                    defaults={'entity_type': 'UNKNOWN'}
                )
                KnowledgeTriple.objects.get_or_create(
                    subject=sub,
                    predicate=item['predicate'],
                    object_entity=obj,
                    user=user,
                    defaults={'source_message': session_obj.messages.first() if session_obj else None}
                )
        except Exception as e:
            print(f"Error extracting graph data: {e}")

    def _create_summary(self, user, text):
        # Generate summary via LLM
        prompt = f"Summarize the key themes and insights from this conversation:\n{text[:20000]}"
        try:
            response = self.client.models.generate_content(
                model=self.extraction_model,
                contents=prompt
            )
            summary_text = response.text.strip()

            # Create HierarchicalSummary object
            HierarchicalSummary.objects.create(
                user=user,
                start_date=user.date_joined,  # Or use session start date
                end_date=user.date_joined,  # Or use session end date
                level='session',
                content=summary_text,
                embedding=None  # Could generate embedding here
            )
        except Exception as e:
            print(f"Error creating summary: {e}")

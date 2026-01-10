import numpy as np
import google.genai as genai
from django.conf import settings
from django.db.models import Q, F
from datetime import datetime, timedelta
import logging
import json

# Import your models
from .models import (
    ChatMessage, Entity, KnowledgeTriple, HierarchicalSummary
)
from core.models import TranscriptEmbedding

logger = logging.getLogger(__name__)

class AdvancedRetrievalService:
    def __init__(self, user, client):
        self.user = user
        self.client = client
        self.embedding_model = "models/text-embedding-004"
        self.llm_model = "gemma-2-27b-it"

    def retrieve(self, user_query: str) -> str:
        """
        Master orchestrator for retrieval.
        1. Decompose Query (Is it Micro, Macro, or Graph?)
        2. Parallel Execution of search strategies.
        3. Rerank and Synthesize.
        """
        # 1. Analyze Intent & Extract Entities from Query
        analysis = self._analyze_query_intent(user_query)

        context_parts = []

        # 2. Execute Strategy based on intent

        # --- PATH A: MICRO / PRECISE (Vector + Keyword) ---
        if analysis['requires_precise_data']:
            # Hybrid Search: Vectors + BM25 Keywords
            precise_hits = self._hybrid_search(
                user_query,
                analysis['keywords'],
                date_range=analysis.get('date_range')
            )
            context_parts.append(f"### PRECISE EXCERPTS:\n{precise_hits}")

        # --- PATH B: GRAPH / RELATIONSHIPS (Entity Linking) ---
        if analysis['entities_detected']:
            graph_data = self._query_knowledge_graph(analysis['entities_detected'])
            if graph_data:
                context_parts.append(f"### KNOWLEDGE GRAPH FACTS:\n{graph_data}")

        # --- PATH C: MACRO / VAST (Summaries) ---
        if analysis['requires_broad_context']:
            macro_summary = self._retrieve_macro_summaries(user_query)
            context_parts.append(f"### LONG-TERM TRENDS/SUMMARIES:\n{macro_summary}")

        return "\n\n".join(context_parts)

    def _analyze_query_intent(self, query: str) -> dict:
        """
        Uses LLM to classify the question type (Micro vs Macro) and extract filters.
        """
        prompt = f"""
        Analyze this user query for retrieval strategy. Return JSON.
        Query: "{query}"

        JSON Format:
        {{
            "requires_precise_data": boolean, (True for specific facts, code, dates)
            "requires_broad_context": boolean, (True for trends, evolution, "summary of all")
            "entities_detected": list[str], (Names, Projects, Topics)
            "keywords": list[str], (Key terms for search)
            "date_range": "YYYY-MM-DD,YYYY-MM-DD" or null
        }}
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp', # Use a fast model for routing
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception:
            # Fallback default
            return {
                "requires_precise_data": True,
                "requires_broad_context": False,
                "entities_detected": [],
                "keywords": query.split()
            }

    def _hybrid_search(self, query: str, keywords: list, date_range=None) -> str:
        """
        Combines Semantic Search (Vector) with Lexical Search (Keyword/BM25).
        """
        # 1. Semantic Search (Vector)
        query_vec = self._get_embedding(query)
        vector_results = []

        # Logic to perform dot product in DB (assuming pgvector) or fetch & sort in python
        # Here we perform Python-side sort for compatibility with the original snippet
        if query_vec:
            # Fetch candidates (optimize by limiting DB fetch to recent or relevancy)
            candidates = TranscriptEmbedding.objects.filter(
                transcript__user=self.user
            ).select_related('segment')

            # Filter by date if applicable
            if date_range:
                # Add date logic here
                pass

            # Calculate Scores
            # (In production, use pgvector cosine distance in the DB query)
            results_with_score = []
            for item in candidates:
                score = np.dot(item.embedding, query_vec)
                results_with_score.append((score, item))

            # Top 10 Vector Hits
            vector_results = sorted(results_with_score, key=lambda x: x[0], reverse=True)[:10]

        # 2. Keyword Search (Precise)
        # Using Django's Q objects for basic OR search, or Postgres SearchQuery for advanced
        keyword_q = Q()
        for k in keywords:
            keyword_q |= Q(segment__text__icontains=k)

        keyword_hits = TranscriptEmbedding.objects.filter(
            keyword_q, transcript__user=self.user
        ).select_related('segment')[:5]

        # 3. Fusion (Reranking/Deduping)
        # Combine vector_results and keyword_hits, remove duplicates
        final_set = {}

        for score, item in vector_results:
            final_set[item.id] = f"[Similarity: {score:.2f}] {item.segment.text}"

        for item in keyword_hits:
            if item.id not in final_set:
                final_set[item.id] = f"[Keyword Match] {item.segment.text}"

        return "\n".join(list(final_set.values()))

    def _query_knowledge_graph(self, entities: list) -> str:
        """
        Retrieves edges connected to detected entities.
        e.g., Query "Sarah" -> Returns "Sarah is HR Manager", "Sarah assigned Project X"
        """
        if not entities:
            return ""

        q_objects = Q()
        for ent in entities:
            q_objects |= Q(name__icontains=ent)

        # Find Entity IDs
        found_entities = Entity.objects.filter(q_objects, user=self.user)

        # Find Triples where these entities are Subject OR Object
        triples = KnowledgeTriple.objects.filter(
            Q(subject__in=found_entities) | Q(object_entity__in=found_entities)
        ).select_related('subject', 'object_entity')

        facts = []
        for t in triples:
            facts.append(f"{t.subject.name} -> {t.predicate} -> {t.object_entity.name}")

        return "\n".join(facts)

    def _retrieve_macro_summaries(self, query: str) -> str:
        """
        Retrieves high-level summaries rather than raw text.
        Used for "How have I changed?" or "Summary of Project X".
        """
        # Embed the query to find semantically relevant *summaries*
        query_vec = self._get_embedding(query)
        if not query_vec: return ""

        # Assume HierarchicalSummary has a vector field (requires pgvector for DB sort)
        # Fetching all summaries for Python sort (Inefficient for huge datasets, okay for snippet)
        summaries = HierarchicalSummary.objects.filter(user=self.user)

        scored = []
        for s in summaries:
            if s.embedding:
                score = np.dot(s.embedding, query_vec)
                scored.append((score, s))

        top_summaries = sorted(scored, key=lambda x: x[0], reverse=True)[:5]

        return "\n".join([f"Timeframe {s[1].start_date.date()}: {s[1].content}" for s in top_summaries])

    def _get_embedding(self, text):
        # ... (Same embedding logic as original) ...
        try:
            return self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                task_type="retrieval_query"
            ).embedding
        except:
            return None

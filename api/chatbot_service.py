import numpy as np
import google.generativeai as genai
from django.conf import settings
from django.db.models import Q
from .models import ChatMessage, ChatMessageEmbedding
from core.models import Transcript, TranscriptEmbedding, TranscriptSegment
import re
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, user):
        self.user = user
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Use a lightweight model for embeddings, standard for chat
        self.embedding_model = "models/text-embedding-004"
        self.chat_model_name = "gemini-2.5-flash"

        self.retrieval_threshold = 0.45  # Stricter threshold to reduce noise
        self.top_k = 10  # Number of segments to retrieve

    def generate_response(self, user_message: str) -> str:
        """Main entry point for generating a response."""
        try:
            # 1. Save User Message immediately
            user_chat_obj = self._save_message(user_message, "user")

            # 2. Check for Specific Commands (Data/Metadata queries)
            # This prevents the LLM from hallucinating lists or dates
            command_response = self._handle_specific_commands(user_message)
            if command_response:
                self._save_message(command_response, "assistant")
                return command_response

            # 3. Perform RAG (Retrieval Augmented Generation)
            context_text = self._build_rag_context(user_message)

            # 4. Generate LLM Response
            system_prompt = self._build_system_prompt()

            # Get recent conversation history (last 5 messages for flow)
            history = self._get_recent_history()

            model = genai.GenerativeModel(
                self.chat_model_name,
                system_instruction=system_prompt
            )

            # Construct the final prompt with context
            final_prompt = (
                f"CONTEXT FROM TRANSCRIPTS AND MEMORY:\n{context_text}\n\n"
                f"CURRENT USER QUESTION: {user_message}"
            )

            chat = model.start_chat(history=history)
            response = chat.send_message(final_prompt)
            answer = response.text.strip()

            # 5. Save Assistant Response
            self._save_message(answer, "assistant")

            return answer

        except Exception as e:
            logger.error(f"Error in ChatbotService: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your request."

    def _handle_specific_commands(self, text: str) -> Optional[str]:
        """
        Handles explicit requests for metadata, lists, or summaries that
        are better served by DB queries than semantic search.
        """
        text = text.lower()

        # Command: List transcripts
        if any(x in text for x in ['list all transcripts', 'show my transcripts', 'list my sessions']):
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')[:20]
            if not transcripts:
                return "You don't have any transcript sessions yet."

            lines = ["Here are your most recent sessions:"]
            for t in transcripts:
                lines.append(f"• **{t.title}** ({t.created_at.strftime('%b %d, %Y %I:%M %p')}) - {t.duration or 'Unknown'} sec")
            return "\n".join(lines)

        # Command: List last X
        match = re.search(r'last (\d+) (transcripts|sessions)', text)
        if match:
            count = int(match.group(1))
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')[:count]
            lines = [f"Here are your last {len(transcripts)} sessions:"]
            for t in transcripts:
                lines.append(f"• {t.title} ({t.created_at.strftime('%Y-%m-%d')})")
            return "\n".join(lines)

        # Command: Summary of specific transcript (Exact or fuzzy title match)
        # Regex looks for: "Summary of 'Meeting X'" or "Summarize Meeting X"
        summary_match = re.search(r'(?:summarize|summary of|summary for) (?:the )?(?:transcript )?["\']?([^"\']+)["\']?', text)
        if summary_match and 'last' not in text:
            query_title = summary_match.group(1).strip()
            # Try exact match first, then icontains
            transcript = Transcript.objects.filter(user=self.user, title__iexact=query_title).first()
            if not transcript:
                transcript = Transcript.objects.filter(user=self.user, title__icontains=query_title).first()

            if transcript:
                return self._generate_transcript_summary(transcript)

        return None

    def _build_rag_context(self, query: str) -> str:
        """Retrieves relevant transcript segments and chat history via Vector Search."""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return ""

        # 1. Search Transcript Segments
        relevant_segments = self._vector_search_transcripts(query_embedding)

        # 2. Search Past Chat History (Long-term memory)
        relevant_chats = self._vector_search_chats(query_embedding)

        context_parts = []

        if relevant_segments:
            context_parts.append("RELEVANT TRANSCRIPT SEGMENTS:")
            for seg in relevant_segments:
                # Format: [Title - Date] Speaker: Text
                meta = f"[{seg['title']} - {seg['date']}]"
                context_parts.append(f"{meta} Speaker {seg['speaker']}: {seg['text']}")

        if relevant_chats:
            context_parts.append("\nRELEVANT PAST CONVERSATIONS:")
            for chat in relevant_chats:
                context_parts.append(f"{chat['role'].upper()}: {chat['content']}")

        if not context_parts:
            return "No specific documents found relevant to this query. Answer based on general knowledge or recent context."

        return "\n".join(context_parts)

    def _vector_search_transcripts(self, query_vec: List[float]) -> List[Dict]:
        """
        Optimized vector search using NumPy matrix multiplication.
        """
        # Fetch all embeddings for this user (optimize this with pgvector in production)
        embeddings_qs = TranscriptEmbedding.objects.filter(
            transcript__user=self.user
        ).select_related('segment', 'transcript').values(
            'segment__id', 'segment__text', 'segment__speaker',
            'transcript__title', 'transcript__created_at', 'embedding'
        )

        if not embeddings_qs:
            return []

        # Convert to separate lists for vectorization
        data_list = list(embeddings_qs)
        vectors = [item['embedding'] for item in data_list]

        if not vectors:
            return []

        # Calculate Cosine Similarity
        # Similarity = (A . B) / (||A|| * ||B||)
        # Note: If embeddings are normalized (Google's usually are), dot product is enough.
        # We assume normalized for speed, otherwise divide by norms.

        query_vec_np = np.array(query_vec)
        matrix = np.array(vectors)

        scores = np.dot(matrix, query_vec_np)

        # Get top K indices
        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > self.retrieval_threshold:
                item = data_list[idx]
                results.append({
                    'text': item['segment__text'],
                    'speaker': item['segment__speaker'],
                    'title': item['transcript__title'],
                    'date': item['transcript__created_at'].strftime('%Y-%m-%d'),
                    'score': scores[idx]
                })

        return results

    def _vector_search_chats(self, query_vec: List[float]) -> List[Dict]:
        """Search past chat messages for context."""
        # Exclude the message just sent
        qs = ChatMessageEmbedding.objects.filter(
            chat_message__user=self.user
        ).exclude(chat_message__content=self.user_current_msg_content if hasattr(self, 'user_current_msg_content') else "").select_related('chat_message')

        if not qs.exists():
            return []

        data = list(qs.values('chat_message__role', 'chat_message__content', 'embedding'))
        vectors = np.array([d['embedding'] for d in data])
        query_vec_np = np.array(query_vec)

        scores = np.dot(vectors, query_vec_np)
        top_indices = np.argsort(scores)[-5:][::-1] # Top 5 chats

        results = []
        for idx in top_indices:
            if scores[idx] > self.retrieval_threshold:
                results.append(data[idx])
        return results

    def _generate_transcript_summary(self, transcript: Transcript) -> str:
        """Summarizes a specific transcript on demand."""
        segments = transcript.segments.all().order_by('start_time')
        full_text = "\n".join([f"{s.speaker}: {s.text}" for s in segments])

        prompt = f"""
        Analyze the following transcript titled "{transcript.title}".
        Provide a concise summary, listing key topics discussed, decisions made, and any action items.

        TRANSCRIPT:
        {full_text[:30000]}  # Truncate to avoid context limits if huge
        """

        model = genai.GenerativeModel(self.chat_model_name)
        response = model.generate_content(prompt)
        return f"**Summary of {transcript.title}:**\n\n{response.text}"

    def _build_system_prompt(self) -> str:
        """
        Constructs a system prompt that gives the LLM 'Meta-Awareness'
        of the user's data environment.
        """
        # Get a list of recent transcript titles to ground the AI
        recent = Transcript.objects.filter(user=self.user).order_by('-created_at')[:5]
        recent_list = ", ".join([f"'{t.title}' ({t.created_at.date()})" for t in recent])

        return (
            f"You are an intelligent assistant for user {self.user.username}. "
            f"Current Date: {datetime.now().strftime('%Y-%m-%d')}. "
            f"The user has recently recorded these transcripts: [{recent_list}]. "
            "Your Goal: Answer questions using the provided Context. "
            "If the answer is found in the Context, cite the transcript title. "
            "If the user asks for a summary or list that you cannot generate from context, "
            "politely suggest they use specific keywords like 'list transcripts'."
        )

    def _get_recent_history(self) -> List[Dict]:
        """Fetch strict chronological history for the LLM session."""
        msgs = ChatMessage.objects.filter(user=self.user).order_by('-timestamp')[:6]
        history = []
        for m in reversed(msgs):
            # Gemini format
            role = "user" if m.role == "user" else "model"
            history.append({"role": role, "parts": [m.content]})
        return history

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _save_message(self, content: str, role: str):
        """Saves message to DB and generates embedding."""
        if role == 'user':
            self.user_current_msg_content = content # store for exclusion logic

        msg = ChatMessage.objects.create(
            user=self.user,
            role=role,
            content=content
        )

        # Generate embedding for future retrieval
        emb = self._get_embedding(content)
        if emb:
            ChatMessageEmbedding.objects.create(chat_message=msg, embedding=emb)

        return msg

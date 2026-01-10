import numpy as np
import google.genai as genai
from django.conf import settings
from django.db.models import Q
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import re

# Import your models.
# Ensure ChatSession exists in your models.py with fields like 'session_id', 'user', and 'summary'
from .models import ChatMessage, ChatMessageEmbedding, ChatSession
from core.models import Transcript, TranscriptEmbedding
from .retrieval_service import AdvancedRetrievalService
from .memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, user, session_id: str):
        """
        Initializes the service for a specific user and chat session.
        
        Args:
            user: The Django User instance.
            session_id (str): Unique identifier for the current chat conversation.
        """
        self.user = user
        self.session_id = session_id
        
        # Initialize the Google GenAI Client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # --- Model Configuration ---
        self.embedding_model = "models/text-embedding-004"
        
        # Main Model: Handles complex logic and user interaction
        self.chat_model_name = "gemma-3-27b-it" 
        
        # Utility Model: Used for background summarization (Fast/Cheap)
        self.summary_model_name = "gemma-3-27b-it"

        # --- Memory Settings ---
        self.buffer_window_size = 6     # Keep last 6 messages raw in context
        
        # --- RAG Settings ---
        self.retrieval_threshold = 0.45 
        self.top_k = 10 

        # Retrieve or create the session object to access persistent summary
        # Assumes ChatSession model has: user, session_id, and summary (TextField)
        self.chat_session, _ = ChatSession.objects.get_or_create(
            user=self.user,
            session_id=self.session_id
        )

        # Initialize Advanced Retrieval Service
        self.retriever = AdvancedRetrievalService(user=self.user, client=self.client)

        # Initialize Memory Consolidator for background processing
        self.consolidator = MemoryConsolidator(client=self.client)

    def generate_response(self, user_message: str) -> str:
        """
        Main entry point.
        1. Saves User Message.
        2. Builds Context (RAG + Session Memory).
        3. Generates AI Response.
        4. Saves AI Response.
        5. Updates Session Summary (Background step).
        """
        try:
            # 1. Save User Message immediately (linked to session)
            self._save_message(user_message, "user")

            # 2. Check for Specific Commands (Data/Metadata queries)
            command_response = self._handle_specific_commands(user_message)
            if command_response:
                self._save_message(command_response, "assistant")
                return command_response

            # 3. Build Contexts
            # A. Advanced RAG Context (External Knowledge from Transcripts + Graph + Summaries)
            advanced_context = self.retriever.retrieve(user_message)

            # B. Conversation Memory (Current Session Only)
            memory_context = self._build_conversation_memory()

            # 4. Construct System Prompt
            system_prompt = self._build_system_prompt()

            # 5. Construct Final Prompt
            final_prompt_parts = [
                f"{system_prompt}\n",

                "### CONTEXT: LONG TERM MEMORY (Summary of this session so far)",
                f"{memory_context['summary'] if memory_context['summary'] else 'No previous context summary available yet.'}",

                "\n### CONTEXT: ADVANCED RETRIEVAL (Precise facts, relationships, and summaries)",
                f"{advanced_context if advanced_context else 'No relevant data found in knowledge base.'}",

                "\n### CONTEXT: RECENT CHAT BUFFER (Immediate History)"
            ]

            # Add the recent chat buffer (User/Model turns)
            for msg in memory_context['buffer']:
                role_label = "User" if msg['role'] == 'user' else "Model"
                final_prompt_parts.append(f"{role_label}: {msg['content']}")

            # Add current question
            final_prompt_parts.append(f"\nUser: {user_message}")
            final_prompt_parts.append("Model:")

            final_prompt = "\n".join(final_prompt_parts)

            # 6. Generate Response using Gemma 3 27B
            response = self.client.models.generate_content(
                model=self.chat_model_name,
                contents=final_prompt,
                config={
                    'temperature': 0.7, 
                    'top_p': 0.95,
                    'top_k': 40,
                }
            )
            
            answer = response.text.strip()

            # 7. Save Assistant Response
            self._save_message(answer, "assistant")

            # 8. POST-GENERATION: Update Summary for the *next* turn
            # This logic happens after we have the answer, preparing memory for the future.
            self._update_session_summary()

            return answer

        except Exception as e:
            logger.error(f"Error in ChatbotService: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your request."

    def _build_conversation_memory(self) -> Dict[str, Any]:
        """
        Retrieves:
        1. The stored summary from the DB (created after the LAST turn).
        2. The raw buffer of recent messages for THIS session.
        """
        # CHANGED: Filter by session=self.chat_session instead of session_id=...
        recent_msgs = ChatMessage.objects.filter(
            user=self.user,
            session=self.chat_session  # Use the object, not the ID string
        ).order_by('-timestamp')[:self.buffer_window_size]

        # Convert to list and reverse to chronological order (oldest -> newest)
        msgs_list = list(reversed(recent_msgs))

        buffer_data = [{'role': m.role, 'content': m.content} for m in msgs_list]

        # Exclude the current user message (it was just saved, but we add it manually to prompt)
        if buffer_data and buffer_data[-1]['role'] == 'user':
            buffer_data.pop()

        return {
            'summary': self.chat_session.summary, # Use the persisted summary field
            'buffer': buffer_data
        }

    def _update_session_summary(self):
        """
        Generates a summary of the current session history and saves it to the ChatSession model.
        This allows the context to "compress" as the conversation grows.
        """
        try:
            # Fetch entire history for this session to ensure continuity
            # In production, you might want to fetch only the last 50 messages + previous summary
            messages = ChatMessage.objects.filter(
                user=self.user,
                session=self.chat_session # CHANGED: Filter by object
            ).order_by('timestamp')

            if messages.count() < 2:
                # Not enough content to summarize meaningfully
                return

            conversation_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])
            
            prompt = (
                "You are a background process updating the memory for a chatbot. "
                "Summarize the following conversation clearly. "
                "Include key facts, user goals, and specific questions asked. "
                "Do not include meta-commentary like 'The conversation started with'. "
                "Just state the facts.\n\n"
                f"Conversation:\n{conversation_text}"
            )

            # Use the lighter model for this background task
            response = self.client.models.generate_content(
                model=self.summary_model_name,
                contents=prompt
            )
            
            new_summary = response.text.strip()

            # Save to Database
            self.chat_session.summary = new_summary
            self.chat_session.save(update_fields=['summary'])

        except Exception as e:
            logger.warning(f"Failed to update session summary: {e}")

    def _build_rag_context(self, query: str) -> str:
        """Retrieves relevant segments from Transcripts."""
        query_embedding = self._get_embedding(query)
        if not query_embedding: return ""

        relevant_segments = self._vector_search_transcripts(query_embedding)
        
        context_parts = []
        if relevant_segments:
            for seg in relevant_segments:
                context_parts.append(f"File: '{seg['title']}' ({seg['date']}) | Speaker {seg['speaker']}: \"{seg['text']}\"")
        
        return "\n".join(context_parts)

    def _vector_search_transcripts(self, query_vec: List[float]) -> List[Dict]:
        """Performs cosine similarity search on Transcript embeddings."""
        embeddings_qs = TranscriptEmbedding.objects.filter(
            transcript__user=self.user
        ).select_related('segment', 'transcript').values(
            'segment__id', 'segment__text', 'segment__speaker',
            'transcript__title', 'transcript__created_at', 'embedding'
        )

        if not embeddings_qs: return []

        data_list = list(embeddings_qs)
        vectors = [item['embedding'] for item in data_list]

        if not vectors: return []

        query_vec_np = np.array(query_vec)
        matrix = np.array(vectors)
        
        # Calculate Dot Product (assuming normalized embeddings)
        scores = np.dot(matrix, query_vec_np)
        
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

    def _handle_specific_commands(self, text: str) -> Optional[str]:
        """
        Handles explicit requests for metadata, lists, summaries, or full text retrieval.
        """
        text = text.lower()

        # --- 1. COMMAND: LIST TRANSCRIPTS ---
        if any(x in text for x in ['list all transcripts', 'show my transcripts', 'list my sessions']):
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')[:20]
            if not transcripts:
                return "You don't have any transcript sessions yet."
            
            lines = ["Here are your most recent sessions:"]
            for t in transcripts:
                lines.append(f"• **{t.title}** ({t.created_at.strftime('%b %d, %Y %I:%M %p')})")
            return "\n".join(lines)

        # --- 2. COMMAND: READ/SHOW FULL TEXT ---
        # Matches: "Show text of...", "Read the transcript...", "Content of..."
        read_match = re.search(r'(?:read|show|display|get|text of|content of) (?:the )?(?:transcript |conversation )?(?:for |with |about )?["\']?([^"\']+)["\']?', text)
        
        # Ensure we don't accidentally trigger on "show my transcripts"
        if read_match and 'list' not in text and 'transcripts' not in text.replace(read_match.group(1), ""):
            raw_target = read_match.group(1).strip()
            clean_query = self._clean_search_query(raw_target)

            if clean_query:
                # Find the transcript
                transcript = Transcript.objects.filter(user=self.user, title__icontains=clean_query).first()
                if transcript:
                    segments = transcript.segments.all().order_by('start_time')
                    if not segments.exists():
                        return f"**Transcript:** {transcript.title}\n*(This transcript is empty)*"
                    
                    lines = [f"**Full Text for '{transcript.title}':**\n"]
                    for s in segments:
                        lines.append(f"**{s.speaker}:** {s.text}")
                    return "\n".join(lines)
                else:
                    return f"I couldn't find a transcript matching '{clean_query}' to read."

        # --- 3. COMMAND: SUMMARIZE ---
        summary_match = re.search(r'(?:summarize|summary of|summary for) (.*)', text)
        
        if summary_match and 'last' not in text:
            raw_target = summary_match.group(1).strip()
            clean_query = self._clean_search_query(raw_target)

            # Logic: If query is empty (e.g. user said "Summarize it" -> cleaned to ""), 
            # assume they want the MOST RECENT transcript.
            if not clean_query:
                transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')[:1]
                if not transcripts.exists():
                     return "You don't have any transcripts to summarize."
                return self._generate_multi_transcript_summary(transcripts)
            else:
                # Specific search
                transcripts = Transcript.objects.filter(user=self.user, title__icontains=clean_query).order_by('-created_at')
                if transcripts.exists():
                    return self._generate_multi_transcript_summary(transcripts)
                else:
                    return f"I couldn't find any transcripts with the title containing '{clean_query}'."

        return None

    def _clean_search_query(self, raw_text: str) -> str:
        """Helper to remove noise words from search queries."""
        stop_words = {
            'my', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'them',
            'conversation', 'conversations', 'session', 'sessions', 'meeting', 'meetings',
            'transcript', 'transcripts', 'chat', 'chats',
            'with', 'about', 'for', 'of', 'both', 'all', 'please', 'can', 'you', 'me'
        }
        tokens = raw_text.split()
        clean_tokens = [t for t in tokens if t not in stop_words]
        return " ".join(clean_tokens).strip()

    def _generate_multi_transcript_summary(self, transcripts) -> str:
        """Helper to summarize full transcripts via RAG/LLM."""
        response_parts = []
        count = transcripts.count()
        response_parts.append(f"Found {count} transcript{'s' if count > 1 else ''} matching your request.\n")

        for transcript in transcripts:
            response_parts.append(f"### {transcript.title} ({transcript.created_at.strftime('%b %d')})")
            segments = transcript.segments.all().order_by('start_time')
            if not segments.exists():
                response_parts.append("*(Empty)*\n")
                continue
            
            # Cap context to avoid token limits
            full_text = "\n".join([f"{s.speaker}: {s.text}" for s in segments])[:20000] 
            
            try:
                prompt = f"Summarize key topics, decisions, and action items for this transcript:\n{full_text}"
                resp = self.client.models.generate_content(
                    model=self.chat_model_name, 
                    contents=prompt
                )
                response_parts.append(resp.text + "\n")
            except Exception:
                response_parts.append("*(Error generating summary)*\n")

        return "\n".join(response_parts)

    def _build_system_prompt(self) -> str:
        recent = Transcript.objects.filter(user=self.user).order_by('-created_at')[:5]
        recent_list = ", ".join([f"'{t.title}' ({t.created_at.date()})" for t in recent])

        return (
            f"You are an intelligent assistant for user {self.user.username}. "
            f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
            f"User's Recent Transcripts available in DB: [{recent_list}]. "
            "Use the provided 'Retrieved Transcripts' to answer factual questions about files. "
            "Use 'Chat Buffer' and 'Memory' to maintain conversation flow."
        )

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                task_type="retrieval_query"
            )
            return result.embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _save_message(self, content: str, role: str):
        """Saves message with link to the current session_id."""
        msg = ChatMessage.objects.create(
            user=self.user,
            session=self.chat_session, # CHANGED: Pass the model instance
            role=role,
            content=content
        )
        
        # Only embed user messages or meaningful assistant responses
        if role == 'user' or len(content) > 50:
            emb = self._get_embedding(content)
            if emb:
                ChatMessageEmbedding.objects.create(chat_message=msg, embedding=emb)
        return msg

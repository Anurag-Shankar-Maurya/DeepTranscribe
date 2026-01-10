import numpy as np
import google.genai as genai
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
        # Initialize the Google GenAI Client (v1.0+ SDK)
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # --- Model Configuration ---
        self.embedding_model = "models/text-embedding-004"
        
        # Main Model: Gemma 3 27B IT (as requested)
        self.chat_model_name = "gemma-3-27b-it" 
        
        # Utility Model: Used for background summarization (Fast/Cheap)
        # We use Flash-Lite here to keep latency low while Gemma handles the complex logic
        self.summary_model_name = "gemma-3-12b-it"

        # --- Memory Settings ---
        self.buffer_window_size = 6     # Keep last 6 messages raw (Short-term memory)
        self.summary_window_lookback = 20 # Look back 20 messages for summary (Long-term memory)
        
        # --- RAG Settings ---
        self.retrieval_threshold = 0.45 
        self.top_k = 10 

    def generate_response(self, user_message: str) -> str:
        """Main entry point for generating a response."""
        try:
            # 1. Save User Message immediately
            self._save_message(user_message, "user")

            # 2. Check for Specific Commands (Data/Metadata queries)
            command_response = self._handle_specific_commands(user_message)
            if command_response:
                self._save_message(command_response, "assistant")
                return command_response

            # 3. Build Contexts
            # A. RAG Context (External Knowledge from Transcripts)
            rag_context = self._build_rag_context(user_message)
            
            # B. Conversation Memory (Internal Context from Chat History)
            # Returns { 'summary': "User previously asked X...", 'buffer': [...] }
            memory_context = self._build_conversation_memory()

            # 4. Construct System Prompt
            system_prompt = self._build_system_prompt()

            # 5. Construct Final Prompt with Explicit Sections
            # Gemma 2/3 instruction models respond well to clear delimiters
            final_prompt_parts = [
                f"{system_prompt}\n",
                
                "### CONTEXT: LONG TERM MEMORY (Summary of past conversation)",
                f"{memory_context['summary'] if memory_context['summary'] else 'No previous context.'}",
                
                "\n### CONTEXT: RETRIEVED TRANSCRIPTS (RAG Data)",
                f"{rag_context if rag_context else 'No specific transcript data found.'}",
                
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
                    'temperature': 0.7, # Balanced creativity
                    'top_p': 0.95,
                    'top_k': 40,
                }
            )
            
            answer = response.text.strip()

            # 7. Save Assistant Response
            self._save_message(answer, "assistant")

            return answer

        except Exception as e:
            logger.error(f"Error in ChatbotService: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your request."

    def _build_conversation_memory(self) -> Dict[str, Any]:
        """
        Splits chat history into:
        1. Buffer: Raw text of the last N messages (High fidelity).
        2. Summary: A generated summary of messages prior to the buffer (Low fidelity).
        """
        total_fetch = self.buffer_window_size + self.summary_window_lookback
        
        # Get messages in reverse chronological order (newest first)
        recent_msgs = ChatMessage.objects.filter(user=self.user).order_by('-timestamp')[:total_fetch]
        
        # Convert to list and reverse to chronological order (oldest first)
        msgs_list = list(reversed(recent_msgs))
        
        if not msgs_list:
            return {'summary': "", 'buffer': []}

        # Slicing
        # The buffer is the last N messages
        buffer_msgs = msgs_list[-self.buffer_window_size:] if len(msgs_list) > self.buffer_window_size else msgs_list
        
        # The items to summarize are everything before the buffer
        to_summarize_msgs = msgs_list[:-self.buffer_window_size] if len(msgs_list) > self.buffer_window_size else []

        # Generate Summary Memory
        summary_text = ""
        if to_summarize_msgs:
            summary_text = self._generate_history_summary(to_summarize_msgs)

        # Format Buffer Memory
        buffer_data = [{'role': m.role, 'content': m.content} for m in buffer_msgs]
        
        # Exclude the current user message (it was just saved, we add it manually to end of prompt)
        if buffer_data and buffer_data[-1]['role'] == 'user':
            buffer_data.pop()

        return {
            'summary': summary_text,
            'buffer': buffer_data
        }

    def _generate_history_summary(self, messages: List[ChatMessage]) -> str:
        """
        Uses a lightweight model to summarize older conversation turns.
        """
        if not messages:
            return ""
            
        conversation_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])
        
        prompt = (
            "Summarize the following conversation history concisely in 3-4 sentences. "
            "Focus strictly on the facts, specific entities discussed, and user questions. "
            "Ignore pleasantries.\n\n"
            f"{conversation_text}"
        )

        try:
            # Use Flash-Lite for fast summarization
            response = self.client.models.generate_content(
                model=self.summary_model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate history summary: {e}")
            return ""

    def _build_rag_context(self, query: str) -> str:
        """RAG Logic: Retrieve Transcript segments."""
        query_embedding = self._get_embedding(query)
        if not query_embedding: return ""

        relevant_segments = self._vector_search_transcripts(query_embedding)
        
        context_parts = []
        if relevant_segments:
            for seg in relevant_segments:
                context_parts.append(f"File: '{seg['title']}' ({seg['date']}) | Speaker {seg['speaker']}: \"{seg['text']}\"")
        
        return "\n".join(context_parts)

    def _vector_search_transcripts(self, query_vec: List[float]) -> List[Dict]:
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
        # Assuming embeddings are normalized
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
        text = text.lower()
        if any(x in text for x in ['list all transcripts', 'show my transcripts', 'list my sessions']):
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')[:20]
            if not transcripts:
                return "You don't have any transcript sessions yet."
            lines = ["Here are your most recent sessions:"]
            for t in transcripts:
                lines.append(f"• **{t.title}** ({t.created_at.strftime('%b %d, %Y %I:%M %p')})")
            return "\n".join(lines)

        summary_match = re.search(r'(?:summarize|summary of|summary for) (.*)', text)
        if summary_match and 'last' not in text:
            raw_target = summary_match.group(1).strip()
            # Stop words to clean up "Summarize the transcript about X"
            stop_words = {'my', 'the', 'a', 'an', 'this', 'session', 'transcript', 'meeting'}
            clean_tokens = [t for t in raw_target.split() if t not in stop_words]
            clean_query = " ".join(clean_tokens).strip()
            
            if not clean_query: return "Please specify a name or topic to summarize."
            
            transcripts = Transcript.objects.filter(user=self.user, title__icontains=clean_query).order_by('-created_at')
            if transcripts.exists():
                return self._generate_multi_transcript_summary(transcripts)
            return f"No transcripts found matching '{clean_query}'."
        return None

    def _generate_multi_transcript_summary(self, transcripts) -> str:
        response_parts = []
        count = transcripts.count()
        response_parts.append(f"Found {count} transcript{'s' if count > 1 else ''} matching your request.\n")

        for transcript in transcripts:
            response_parts.append(f"### {transcript.title} ({transcript.created_at.strftime('%b %d')})")
            segments = transcript.segments.all().order_by('start_time')
            if not segments.exists():
                response_parts.append("*(Empty)*\n")
                continue
            
            full_text = "\n".join([f"{s.speaker}: {s.text}" for s in segments])[:20000] # Cap context
            
            try:
                # Use the main intelligent model for detailed transcript summarization
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
            f"You are an intelligent assistant for user {self.user.username}."
            f"Today is {datetime.now().strftime('%Y-%m-%d')}."
            f"User's Recent Transcripts: [{recent_list}]."
            "Your inputs include 'Long Term Memory' (summaries of past chat), "
            "'Retrieved Transcripts' (RAG data), and 'Chat Buffer' (immediate conversation)."
            "Prioritize the Retrieved Transcripts for factual questions about files. "
            "Use Chat Buffer to understand immediate context."
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
        msg = ChatMessage.objects.create(user=self.user, role=role, content=content)
        # Only embed user messages or meaningful assistant responses for future search
        if role == 'user' or len(content) > 50:
            emb = self._get_embedding(content)
            if emb:
                ChatMessageEmbedding.objects.create(chat_message=msg, embedding=emb)
        return msg
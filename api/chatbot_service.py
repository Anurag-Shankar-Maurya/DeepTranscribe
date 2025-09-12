import numpy as np
import google.generativeai as genai
from django.conf import settings
from .models import ChatMessage, ChatMessageEmbedding
from core.models import Transcript, TranscriptEmbedding, TranscriptSegment
from collections import Counter
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

class ChatbotService:
    def __init__(self, user):
        self.user = user
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.system_instruction = (
            "You are a highly intelligent and helpful assistant with access to all transcript data and chat history. "
            "You can perform content-specific retrieval, summarization, Q&A, contextual understanding, comparative analysis, "
            "analytics, entity recognition, timeline reconstruction, and recommendation generation based on the complete database. "
            "Always answer based on the transcript and chat data available. Provide detailed, accurate, and context-aware responses. "
            "If information is not available, politely inform the user. Use the transcript segments and chat messages as your knowledge base."
        )
        self.retrieval_threshold = 0.3  # Minimum similarity score for considering a match

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text using Gemini's API"""
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            return response['embedding']
        except Exception as e:
            print(f"Gemini API error during embedding generation: {e}")
            return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_relevant_content(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve relevant content from both chat messages and transcripts
        Returns tuple of (relevant_chat_messages, relevant_transcript_segments)
        """
        query_embedding = self.get_embeddings(query)
        if query_embedding is None:
            return ([], []) # Handle case where embedding generation fails

        # Get relevant chat messages
        chat_messages = []
        chat_embeddings_data = [] # Store embeddings directly for similarity calculation
        for msg in ChatMessage.objects.filter(user=self.user).order_by('timestamp'):
            try:
                # Assuming ChatMessageEmbedding stores embeddings compatible with Gemini
                emb_obj = msg.embedding
                if emb_obj and emb_obj.embedding:
                    chat_embeddings_data.append({'embedding': emb_obj.embedding, 'message': msg})
            except ChatMessageEmbedding.DoesNotExist:
                continue

        chat_similarities = []
        if chat_embeddings_data:
            chat_similarities = [self.cosine_similarity(query_embedding, item['embedding']) for item in chat_embeddings_data]
        
        # Get relevant transcript segments
        transcript_segments = []
        transcript_embeddings_data = [] # Store embeddings directly for similarity calculation
        for te in TranscriptEmbedding.objects.filter(
            transcript__user=self.user
        ).select_related('transcript', 'segment'):
            if te.segment and te.embedding: # Ensure segment and embedding exist
                transcript_embeddings_data.append({'embedding': te.embedding, 'segment': te.segment, 'transcript': te.transcript})

        transcript_similarities = []
        if transcript_embeddings_data:
            transcript_similarities = [self.cosine_similarity(query_embedding, item['embedding']) for item in transcript_embeddings_data]
        
        # Combine and sort all content by similarity
        all_content = []
        for i, item in enumerate(chat_embeddings_data):
            all_content.append({
                'content': {
                    'id': item['message'].id,
                    'role': item['message'].role,
                    'content': item['message'].content,
                    'timestamp': item['message'].timestamp,
                    'type': 'chat'
                },
                'similarity': chat_similarities[i] if i < len(chat_similarities) else 0
            })
        
        for i, item in enumerate(transcript_embeddings_data):
            all_content.append({
                'content': {
                    'id': item['segment'].id,
                    'text': item['segment'].text,
                    'speaker': item['segment'].speaker,
                    'start_time': item['segment'].start_time,
                    'end_time': item['segment'].end_time,
                    'transcript_id': item['transcript'].id,
                    'transcript_title': item['transcript'].title,
                    'transcript_date': item['transcript'].created_at,
                    'type': 'transcript'
                },
                'similarity': transcript_similarities[i] if i < len(transcript_similarities) else 0
            })
        
        # Sort by similarity and filter by threshold
        all_content.sort(key=lambda x: x['similarity'], reverse=True)
        filtered_content = [x['content'] for x in all_content if x['similarity'] >= self.retrieval_threshold]
        
        # Separate back into chat and transcript content
        relevant_chats = [x for x in filtered_content if x['type'] == 'chat']
        relevant_transcripts = [x for x in filtered_content if x['type'] == 'transcript']
        
        return (relevant_chats[:top_k], relevant_transcripts[:top_k])

    def analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'positive', 'happy', 'success', 'excellent', 'awesome']
        negative_words = ['bad', 'problem', 'issue', 'negative', 'fail', 'terrible', 'worst']
        text_lower = text.lower()
        pos_count = sum(text_lower.count(w) for w in positive_words)
        neg_count = sum(text_lower.count(w) for w in negative_words)
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities (names, organizations) from text"""
        # Improved regex to capture more entity types
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
        return list(set(entities))

    def get_speaker_activity(self) -> Dict[str, int]:
        """Get speaker activity statistics"""
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        speaker_counts = Counter(seg.speaker for seg in segments if seg.speaker is not None)
        return dict(speaker_counts)

    def get_topic_frequency(self, query: str = None) -> Dict[str, int]:
        """Get frequency of topics across transcripts"""
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        if query:
            query_embedding = self.get_embeddings(query)
            if query_embedding is None: return {query: 0} # Handle embedding failure
            topic_segments = []
            for te in TranscriptEmbedding.objects.filter(transcript__user=self.user).select_related('segment'):
                if te.embedding: # Ensure embedding exists
                    sim = self.cosine_similarity(query_embedding, te.embedding)
                    if sim >= self.retrieval_threshold and te.segment:
                        topic_segments.append(te.segment.text)
            return {query: len(topic_segments)}
        else:
            # For general topic frequency, we'd ideally use topic modeling, but this is a simplified version
            all_text = " ".join(seg.text for seg in segments)
            entities = self.extract_entities(all_text)
            return Counter(entities)

    def generate_timeline(self, topic: str = None) -> List[Dict]:
        """Generate timeline of events or mentions of a specific topic"""
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        if topic:
            query_embedding = self.get_embeddings(topic)
            if query_embedding is None: return [] # Handle embedding failure
            relevant_segments = []
            for te in TranscriptEmbedding.objects.filter(transcript__user=self.user).select_related('segment'):
                if te.embedding: # Ensure embedding exists
                    sim = self.cosine_similarity(query_embedding, te.embedding)
                    if sim >= self.retrieval_threshold and te.segment:
                        relevant_segments.append(te.segment)
            segments = relevant_segments
        
        timeline = []
        for seg in segments.order_by('start_time'):
            timeline.append({
                'time': seg.start_time,
                'text': seg.text,
                'speaker': seg.speaker,
                'transcript': seg.transcript.title,
                'date': seg.transcript.created_at
            })
        return timeline

    def generate_summary(self, transcript_id: int = None) -> str:
        """Generate summary for a specific transcript or all transcripts"""
        if transcript_id:
            try:
                transcript = Transcript.objects.get(id=transcript_id, user=self.user)
                segments = transcript.segments.all().order_by('start_time')
                full_text = "\n".join([segment.text for segment in segments])
                summary_prompt = f"Please provide a detailed and comprehensive summary of the following transcript session:\n{full_text}"
            except Transcript.DoesNotExist:
                return "Transcript not found."
        else:
            # Summarize all transcripts
            transcripts = Transcript.objects.filter(user=self.user)
            summary_texts = []
            for t in transcripts:
                segments = t.segments.all().order_by('start_time')
                full_text = "\n".join([segment.text for segment in segments])
                summary_texts.append(f"Transcript: {t.title} (Date: {t.created_at})\nContent: {full_text[:1000]}...")
            
            full_text = "\n\n".join(summary_texts)
            summary_prompt = f"Please provide a comprehensive summary of all transcript sessions, highlighting key themes, decisions, and action items:\n{full_text}"
        
        # Use Gemini chat completion
        try:
            model = genai.GenerativeModel(
                'gemini-2.5-flash-lite',
                system_instruction="You are a helpful assistant specialized in summarizing transcript sessions."
            )
            response = model.generate_content([
                {"role": "user", "parts": [{"text": summary_prompt}]}
            ])
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error during summary generation: {e}")
            return "An error occurred while generating the summary."

    def generate_response(self, user_message: str) -> str:
        """Main method to generate response to user query"""
        # First check for specific commands
        response = self._handle_specific_commands(user_message)
        if response:
            return response
        
        # If no specific command matched, perform general question answering
        return self._answer_general_question(user_message)

    def _handle_specific_commands(self, user_message: str) -> Optional[str]:
        """Handle specific command patterns"""
        # List all transcripts
        if re.search(r'list all transcripts?', user_message, re.IGNORECASE):
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')
            if not transcripts.exists():
                return "No transcript sessions found."
            return "Transcript sessions:\n" + "\n".join(
                f"{idx+1}. {t.title} (created at {t.created_at.strftime('%Y-%m-%d %H:%M:%S')})"
                for idx, t in enumerate(transcripts))
        
        # List transcripts within time period
        time_match = re.search(r'list transcripts? (?:within|in) last (\d+) (days?|weeks?|months?)', user_message, re.IGNORECASE)
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2).rstrip('s')
            delta = timedelta(
                days=num if unit == 'day' else num*7 if unit == 'week' else num*30
            )
            since_date = datetime.now() - delta
            transcripts = Transcript.objects.filter(
                user=self.user, 
                created_at__gte=since_date
            ).order_by('-created_at')
            if not transcripts.exists():
                return f"No transcripts found within the last {num} {unit}{'s' if num > 1 else ''}."
            return f"Transcripts within last {num} {unit}{'s' if num > 1 else ''}:\n" + "\n".join(
                f"{idx+1}. {t.title} ({t.created_at.strftime('%Y-%m-%d')})"
                for idx, t in enumerate(transcripts))
        
        # Get summary of a specific transcript
        summary_match = re.search(r'(?:summary|summarize) (?:of|for) (?:transcript|session) (?:named|called)? ?(["\']?)(.*?)\1', 
                                user_message, re.IGNORECASE)
        if summary_match:
            transcript_name = summary_match.group(2).strip()
            try:
                transcript = Transcript.objects.get(user=self.user, title__iexact=transcript_name)
                return self.generate_summary(transcript.id)
            except Transcript.DoesNotExist:
                return f"I could not find a transcript session named '{transcript_name}'."
        
        # Get speaker activity
        if re.search(r'(who|which speaker) (talked|spoke) (most|more|the most)', user_message, re.IGNORECASE):
            activity = self.get_speaker_activity()
            if not activity:
                return "No speaker activity data available."
            most_active = max(activity.items(), key=lambda x: x[1])
            return f"Speaker {most_active[0]} talked the most with {most_active[1]} segments."
        
        # Sentiment analysis
        if re.search(r'sentiment (analysis|overview|summary)', user_message, re.IGNORECASE):
            segments = TranscriptSegment.objects.filter(transcript__user=self.user)
            sentiments = [self.analyze_sentiment(seg.text) for seg in segments]
            sentiment_counts = Counter(sentiments)
            return (
                "Sentiment analysis results:\n"
                f"Positive: {sentiment_counts.get('positive', 0)}\n"
                f"Negative: {sentiment_counts.get('negative', 0)}\n"
                f"Neutral: {sentiment_counts.get('neutral', 0)}"
            )
        
        # Entity extraction
        if re.search(r'(list|show) (important )?(names|entities|topics)', user_message, re.IGNORECASE):
            entities = self.get_topic_frequency()
            if not entities:
                return "No important entities found."
            return "Important entities mentioned (with frequency):\n" + "\n".join(
                f"{entity}: {count}" for entity, count in entities.most_common(10))
        
        # Timeline generation
        timeline_match = re.search(r'(?:timeline|when was) (?:of|for) ?(["\']?)(.*?)\1', user_message, re.IGNORECASE)
        if timeline_match:
            topic = timeline_match.group(2).strip()
            timeline = self.generate_timeline(topic)
            if not timeline:
                return f"No timeline information found for '{topic}'."
            
            # Format timeline entries
            formatted_entries = []
            for entry in timeline[:5]:  # Limit to 5 entries for brevity
                time_str = str(timedelta(seconds=int(entry['time'])))
                formatted_entries.append(
                    f"At {time_str} in '{entry['transcript']}': {entry['text'][:100]}..."
                )
            return f"Timeline for '{topic}':\n" + "\n".join(formatted_entries)
        
        # Topic frequency
        freq_match = re.search(r'how many times (?:was|were) (.*?) mentioned', user_message, re.IGNORECASE)
        if freq_match:
            topic = freq_match.group(1).strip()
            freq = self.get_topic_frequency(topic)
            return f"'{topic}' was mentioned {freq.get(topic, 0)} times across all transcripts."
        
        return None

    def _answer_general_question(self, user_message: str) -> str:
        """Handle general questions using vector search and context"""
        # Store user message and get embedding
        user_chat_msg = ChatMessage.objects.create(
            user=self.user,
            role='user',
            content=user_message
        )
        user_embedding = self.get_embeddings(user_message)
        if user_embedding is None:
            return "I'm sorry, I encountered an error generating embeddings for your message. Please try again later."

        ChatMessageEmbedding.objects.create(
            chat_message=user_chat_msg,
            embedding=user_embedding
        )

        # Get relevant content from both chats and transcripts
        relevant_chats, relevant_transcripts = self.get_relevant_content(user_message)

        # Prepare context for the LLM
        messages = []

        # Add system instruction as the first message
        messages.append({"role": "model", "parts": [{"text": self.system_instruction}]})

        # Add relevant chat history
        for chat in relevant_chats:
            role = 'model' if chat['role'] == 'assistant' else 'user'
            messages.append({"role": role, "parts": [{"text": chat['content']}]})

        # Construct a single user message with all context
        context_string = ""
        if relevant_transcripts:
            context_string = "Here is some relevant context from your transcripts:\n"
            for seg in relevant_transcripts:
                context_string += (
                    f"- From '{seg['transcript_title']}' (Speaker {seg['speaker']} at {seg['start_time']}s): "
                    f'"{seg["text"]}"\n'
                )
            context_string += "\n"

        final_user_message = f"{context_string}My question is: {user_message}"
        messages.append({"role": "user", "parts": [{"text": final_user_message}]})

        # Generate response using Gemini chat completion
        try:
            model = genai.GenerativeModel(
                'gemini-2.5-flash-lite',
            )
            response = model.generate_content(messages)
            answer = response.text.strip()
        except Exception as e:
            print(f"Gemini API error during general question answering: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again later."

        # Store assistant response
        assistant_chat_msg = ChatMessage.objects.create(
            user=self.user,
            role='assistant',
            content=answer
        )
        assistant_embedding = self.get_embeddings(answer)
        if assistant_embedding is None:
            print("Warning: Could not generate embedding for assistant's response.")
        else:
            ChatMessageEmbedding.objects.create(
                chat_message=assistant_chat_msg,
                embedding=assistant_embedding
            )

        return answer

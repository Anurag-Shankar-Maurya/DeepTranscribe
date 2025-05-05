import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from django.conf import settings
from .models import ChatMessage
from core.models import Transcript, TranscriptSegment
from collections import Counter
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
import json

class ChatbotService:
    def __init__(self, user):
        self.user = user
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize Pinecone client instance
        try:
            self.pinecone_client = Pinecone(
                api_key=settings.PINECONE_API_KEY
            )
        except AttributeError as e:
            raise RuntimeError("PINECONE_API_KEY must be set in Django settings or environment variables.") from e
        
        # Connect to Pinecone index - only create if it doesn't exist
        self.index_name = settings.PINECONE_INDEX_NAME
        existing_indexes = self.pinecone_client.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=4096,  # Default dimension for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
        
        self.pinecone_index = self.pinecone_client.Index(self.index_name)
        
        self.system_instruction = (
            "You are a highly intelligent and helpful assistant with access to all transcript data and chat history. "
            "You can perform content-specific retrieval, summarization, Q&A, contextual understanding, comparative analysis, "
            "analytics, entity recognition, timeline reconstruction, and recommendation generation based on the complete database. "
            "Always answer based on the transcript and chat data available. Provide detailed, accurate, and context-aware responses. "
            "If information is not available, politely inform the user. Use the transcript segments and chat messages as your knowledge base."
        )
        self.retrieval_threshold = 0.1  # Minimum similarity score for considering a match
        self.conversation_memory = []  # Store recent conversation for context

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text using OpenAI's API"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_relevant_content(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve relevant content from both chat messages and transcripts using Pinecone
        Returns tuple of (relevant_chat_messages, relevant_transcript_segments)
        """
        query_embedding = self.get_embeddings(query)
        
        # Query Pinecone with user-specific filter
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Get extra to filter by type
            filter={
                "$or": [
                    {"type": "chat", "user_id": str(self.user.id)},
                    {"type": "transcript", "user_id": str(self.user.id)}
                ]
            },
            include_metadata=True
        )
        
        # Process results
        relevant_chats = []
        relevant_transcripts = []
        
        for match in results.matches:
            if match.score < self.retrieval_threshold:
                continue
                
            metadata = match.metadata
            if metadata['type'] == 'chat':
                relevant_chats.append({
                    'id': match.id,
                    'role': metadata.get('role'),
                    'content': metadata.get('content'),
                    'timestamp': metadata.get('timestamp'),
                    'type': 'chat'
                })
            elif metadata['type'] == 'transcript':
                relevant_transcripts.append({
                    'id': match.id,
                    'text': metadata.get('text'),
                    'speaker': metadata.get('speaker'),
                    'start_time': metadata.get('start_time'),
                    'end_time': metadata.get('end_time'),
                    'transcript_id': metadata.get('transcript_id'),
                    'transcript_title': metadata.get('transcript_title'),
                    'transcript_date': metadata.get('transcript_date'),
                    'type': 'transcript'
                })
        
        return (relevant_chats[:top_k], relevant_transcripts[:top_k])

    def store_chat_message(self, message_id: str, role: str, content: str):
        """Store chat message in Pinecone with metadata"""
        embedding = self.get_embeddings(content)
        
        self.pinecone_index.upsert(
            vectors=[{
                'id': message_id,
                'values': embedding,
                'metadata': {
                    'type': 'chat',
                    'user_id': str(self.user.id),
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                }
            }]
        )
        
        # Add to conversation memory for context
        self.conversation_memory.append({"role": role, "content": content})
        # Keep only last 10 messages in memory
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]

    def store_transcript_segment(self, segment_id: str, transcript_id: str, text: str, 
                               speaker: str, start_time: float, end_time: float):
        """Store transcript segment in Pinecone with metadata"""
        embedding = self.get_embeddings(text)
        
        transcript = Transcript.objects.get(id=transcript_id)
        
        self.pinecone_index.upsert(
            vectors=[{
                'id': segment_id,
                'values': embedding,
                'metadata': {
                    'type': 'transcript',
                    'user_id': str(self.user.id),
                    'text': text,
                    'speaker': speaker,
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript_id': transcript_id,
                    'transcript_title': transcript.title,
                    'transcript_date': transcript.created_at.isoformat()
                }
            }]
        )

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
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
        return list(set(entities))

    def get_speaker_activity(self) -> Dict[str, int]:
        """Get speaker activity statistics"""
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        speaker_counts = Counter(seg.speaker for seg in segments if seg.speaker is not None)
        return dict(speaker_counts)

    def get_topic_frequency(self, query: str = None) -> Dict[str, int]:
        """Get frequency of topics across transcripts"""
        if query:
            query_embedding = self.get_embeddings(query)
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=100,
                filter={
                    "type": "transcript",
                    "user_id": str(self.user.id)
                },
                include_metadata=True
            )
            return {query: len([m for m in results.matches if m.score >= self.retrieval_threshold])}
        else:
            segments = TranscriptSegment.objects.filter(transcript__user=self.user)
            all_text = " ".join(seg.text for seg in segments)
            entities = self.extract_entities(all_text)
            return Counter(entities)

    def generate_timeline(self, topic: str = None) -> List[Dict]:
        """Generate timeline of events or mentions of a specific topic"""
        if topic:
            query_embedding = self.get_embeddings(topic)
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=100,
                filter={
                    "type": "transcript",
                    "user_id": str(self.user.id)
                },
                include_metadata=True
            )
            
            timeline = []
            for match in results.matches:
                if match.score >= self.retrieval_threshold:
                    metadata = match.metadata
                    timeline.append({
                        'time': metadata['start_time'],
                        'text': metadata['text'],
                        'speaker': metadata['speaker'],
                        'transcript': metadata['transcript_title'],
                        'date': metadata['transcript_date']
                    })
            
            # Sort by time
            timeline.sort(key=lambda x: x['time'])
            return timeline
        else:
            segments = TranscriptSegment.objects.filter(transcript__user=self.user).order_by('start_time')
            return [{
                'time': seg.start_time,
                'text': seg.text,
                'speaker': seg.speaker,
                'transcript': seg.transcript.title,
                'date': seg.transcript.created_at
            } for seg in segments]

    def generate_summary(self, transcript_id: Union[int, str] = None) -> str:
        """Generate summary for a specific transcript or all transcripts"""
        if transcript_id:
            try:
                # Check if transcript_id is a string (title) or an integer (ID)
                if isinstance(transcript_id, str):
                    transcript = Transcript.objects.get(title__iexact=transcript_id, user=self.user)
                else:
                    transcript = Transcript.objects.get(id=transcript_id, user=self.user)
                    
                segments = transcript.segments.all().order_by('start_time')
                full_text = "\n".join([segment.text for segment in segments])
                summary_prompt = f"Please provide a detailed and comprehensive summary of the following transcript session:\n{full_text}"
            except Transcript.DoesNotExist:
                return "Transcript not found."
        else:
            transcripts = Transcript.objects.filter(user=self.user)
            summary_texts = []
            for t in transcripts:
                segments = t.segments.all().order_by('start_time')
                full_text = "\n".join([segment.text for segment in segments])
                summary_texts.append(f"Transcript: {t.title} (Date: {t.created_at})\nContent: {full_text[:1000]}...")
            
            full_text = "\n\n".join(summary_texts)
            summary_prompt = f"Please provide a comprehensive summary of all transcript sessions, highlighting key themes, decisions, and action items:\n{full_text}"
        
        summary_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in summarizing transcript sessions."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return summary_response.choices[0].message.content.strip()

    def list_all_transcripts(self, time_period: Optional[Dict[str, int]] = None) -> str:
        """List all transcripts, optionally filtered by time period"""
        if time_period:
            num = time_period.get('num', 30)
            unit = time_period.get('unit', 'day')
            delta = timedelta(
                days=num if unit == 'day' else num*7 if unit == 'week' else num*30
            )
            since_date = datetime.now() - delta
            transcripts = Transcript.objects.filter(
                user=self.user, 
                created_at__gte=since_date
            ).order_by('-created_at')
            
            period_str = f"{num} {unit}{'s' if num > 1 else ''}"
            if not transcripts.exists():
                return f"No transcripts found within the last {period_str}."
                
            result = f"Transcripts within last {period_str}:\n"
        else:
            transcripts = Transcript.objects.filter(user=self.user).order_by('-created_at')
            if not transcripts.exists():
                return "No transcript sessions found."
                
            result = "Transcript sessions:\n"
            
        # Add transcript details
        for idx, t in enumerate(transcripts):
            result += f"{idx+1}. {t.title} (created at {t.created_at.strftime('%Y-%m-%d %H:%M:%S')})\n"
            
        return result.strip()

    def list_transcript_segments(self, transcript_name: str, limit: int = 10) -> str:
        """List all segments in a specific transcript"""
        try:
            transcript = Transcript.objects.get(user=self.user, title__iexact=transcript_name)
            segments = transcript.segments.all().order_by('start_time')
            
            if not segments.exists():
                return f"No segments found in transcript '{transcript_name}'."
                
            result = f"Segments in '{transcript_name}':\n"
            for idx, seg in enumerate(segments[:limit]):
                time_str = str(timedelta(seconds=int(seg.start_time)))
                result += f"{idx+1}. [{time_str}] {seg.speaker}: {seg.text[:100]}...\n"
                
            if segments.count() > limit:
                result += f"\n... and {segments.count() - limit} more segments."
                
            return result.strip()
        except Transcript.DoesNotExist:
            return f"Transcript '{transcript_name}' not found."

    def get_transcript_info(self, transcript_name: str) -> str:
        """Get detailed information about a specific transcript"""
        try:
            transcript = Transcript.objects.get(user=self.user, title__iexact=transcript_name)
            segments = transcript.segments.all()
            speakers = Counter(seg.speaker for seg in segments if seg.speaker is not None)
            
            duration = 0
            if segments.exists():
                max_time = segments.order_by('-end_time').first().end_time
                duration = max_time
            
            result = (
                f"Transcript Info: {transcript.title}\n"
                f"Created: {transcript.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Duration: {str(timedelta(seconds=int(duration)))}\n"
                f"Total segments: {segments.count()}\n"
                f"Speakers: {', '.join(speakers.keys())}\n"
                f"Most active speaker: {max(speakers.items(), key=lambda x: x[1])[0] if speakers else 'None'}"
            )
            
            return result
        except Transcript.DoesNotExist:
            return f"Transcript '{transcript_name}' not found."

    def search_transcripts(self, query: str, limit: int = 5) -> str:
        """Search through transcript content using semantic search"""
        _, relevant_transcripts = self.get_relevant_content(query, top_k=limit)
        
        if not relevant_transcripts:
            return f"No results found for '{query}'."
            
        result = f"Search results for '{query}':\n"
        for idx, segment in enumerate(relevant_transcripts):
            time_str = str(timedelta(seconds=int(segment['start_time'])))
            result += (
                f"{idx+1}. In '{segment['transcript_title']}' at {time_str}\n"
                f"   {segment['speaker']}: {segment['text']}\n\n"
            )
            
        return result.strip()

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
        user_message_lower = user_message.lower()
        
        # List all transcripts
        if re.search(r'list all transcripts?', user_message_lower):
            return self.list_all_transcripts()
        
        # List transcripts within time period
        time_match = re.search(r'list transcripts? (?:within|in) last (\d+) (days?|weeks?|months?)', user_message_lower)
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2).rstrip('s')
            return self.list_all_transcripts({'num': num, 'unit': unit})
        
        # List segments in transcript
        segments_match = re.search(r'list (?:all )?(?:segments?|talks?|contents?) (?:in|of|for) (?:transcript|session) (?:named|called)? ?(["\']?)(.*?)\1', 
                                user_message_lower)
        if segments_match:
            transcript_name = segments_match.group(2).strip()
            return self.list_transcript_segments(transcript_name)
        
        # Get transcript information
        info_match = re.search(r'(?:info|information|details) (?:about|on|for) (?:transcript|session) (?:named|called)? ?(["\']?)(.*?)\1', 
                            user_message_lower)
        if info_match:
            transcript_name = info_match.group(2).strip()
            return self.get_transcript_info(transcript_name)
        
        # Get summary of a specific transcript
        summary_match = re.search(r'(?:summary|summarize) (?:of|for) (?:transcript|session) (?:named|called)? ?(["\']?)(.*?)\1', 
                                user_message_lower)
        if summary_match:
            transcript_name = summary_match.group(2).strip()
            return self.generate_summary(transcript_name)
        
        # Search transcripts
        search_match = re.search(r'search (?:for |about )?(.*?)(?:\s|$)', user_message_lower)
        if search_match:
            query = search_match.group(1).strip()
            return self.search_transcripts(query)
        
        # Get speaker activity
        if re.search(r'(who|which speaker) (talked|spoke) (most|more|the most)', user_message_lower):
            activity = self.get_speaker_activity()
            if not activity:
                return "No speaker activity data available."
            most_active = max(activity.items(), key=lambda x: x[1])
            return f"Speaker {most_active[0]} talked the most with {most_active[1]} segments."
        
        # Sentiment analysis
        if re.search(r'sentiment (analysis|overview|summary)', user_message_lower):
            segments = TranscriptSegment.objects.filter(transcript__user=self.user)
            if not segments.exists():
                return "No transcript data available for sentiment analysis."
                
            sentiments = [self.analyze_sentiment(seg.text) for seg in segments]
            sentiment_counts = Counter(sentiments)
            return (
                "Sentiment analysis results:\n"
                f"Positive: {sentiment_counts.get('positive', 0)}\n"
                f"Negative: {sentiment_counts.get('negative', 0)}\n"
                f"Neutral: {sentiment_counts.get('neutral', 0)}"
            )
        
        # Entity extraction
        if re.search(r'(list|show) (important )?(names|entities|topics)', user_message_lower):
            entities = self.get_topic_frequency()
            if not entities:
                return "No important entities found."
            return "Important entities mentioned (with frequency):\n" + "\n".join(
                f"{entity}: {count}" for entity, count in entities.most_common(10))
        
        # Timeline generation
        timeline_match = re.search(r'(?:timeline|when was) (?:of|for) ?(["\']?)(.*?)\1', user_message_lower)
        if timeline_match:
            topic = timeline_match.group(2).strip()
            timeline = self.generate_timeline(topic)
            if not timeline:
                return f"No timeline information found for '{topic}'."
            
            formatted_entries = []
            for entry in timeline[:5]:  # Limit to 5 entries for brevity
                time_str = str(timedelta(seconds=int(entry['time'])))
                formatted_entries.append(
                    f"At {time_str} in '{entry['transcript']}': {entry['text'][:100]}..."
                )
            return f"Timeline for '{topic}':\n" + "\n".join(formatted_entries)
        
        # Topic frequency
        freq_match = re.search(r'how many times (?:was|were) (.*?) mentioned', user_message_lower)
        if freq_match:
            topic = freq_match.group(1).strip()
            freq = self.get_topic_frequency(topic)
            return f"'{topic}' was mentioned {freq.get(topic, 0)} times across all transcripts."
        
        return None

    def _answer_general_question(self, user_message: str) -> str:
        """Handle general questions using vector search and context"""
        # Store user message in Django DB and Pinecone
        user_chat_msg = ChatMessage.objects.create(
            user=self.user, 
            role='user', 
            content=user_message
        )
        self.store_chat_message(str(user_chat_msg.id), 'user', user_message)
        
        # Get relevant content from both chats and transcripts
        relevant_chats, relevant_transcripts = self.get_relevant_content(user_message)
        
        # Prepare context for the LLM
        messages = [{"role": "system", "content": self.system_instruction}]
        
        # Add conversation memory for context
        for msg in self.conversation_memory:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add relevant transcript segments as context
        for seg in relevant_transcripts:
            context = (
                f"Transcript Context from '{seg['transcript_title']}' (Speaker {seg['speaker']} at {seg['start_time']}s): "
                f"{seg['text']}"
            )
            messages.append({"role": "system", "content": context})
        
        # Add relevant chat history
        for chat in relevant_chats:
            if {"role": chat['role'], "content": chat['content']} not in self.conversation_memory:
                messages.append({"role": chat['role'], "content": chat['content']})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        
        # Store assistant response
        assistant_chat_msg = ChatMessage.objects.create(
            user=self.user, 
            role='assistant', 
            content=answer
        )
        self.store_chat_message(str(assistant_chat_msg.id), 'assistant', answer)
        
        return answer
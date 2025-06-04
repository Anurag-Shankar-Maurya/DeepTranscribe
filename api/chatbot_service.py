import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from django.conf import settings
from django.db.models import Q
from .models import ChatMessage
from core.models import Transcript
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union
import uuid
import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, user):
        self.user = user
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.pinecone_client = None
        self.pinecone_index = None
        self.index_name = settings.PINECONE_INDEX_NAME
        self.retrieval_threshold = 0.0  # Allow all matches
        self.conversation_memory = []
        self._initialize_pinecone()
        
        self.system_instruction = (
            "You are an intelligent assistant with access to user-specific transcript data and chat history. "
            "Use the provided context to generate accurate, relevant, and concise responses. "
            "If no relevant information is available, clearly state so and provide a helpful response."
        )

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index with error handling"""
        try:
            self.pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
            existing_indexes = self.pinecone_client.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
                )
            
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Pinecone index {self.index_name} initialized. Stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise RuntimeError("Failed to initialize Pinecone client.") from e

    def check_transcript_storage(self, top_k: int = 50) -> List[Dict]:
        """Check stored transcripts in Pinecone for debugging and validate metadata"""
        try:
            results = self.pinecone_index.query(
                vector=[0] * 1536,  # Dummy vector to fetch data
                top_k=top_k,
                filter={"user_id": str(self.user.id), "type": "transcript"},
                include_metadata=True
            )
            items = []
            for match in results.matches:
                metadata = match.metadata
                # Validate required fields
                if not metadata.get('timestamp'):
                    logger.warning(f"Transcript {match.id} has missing or None timestamp: {metadata}")
                if not metadata.get('transcript_title'):
                    logger.warning(f"Transcript {match.id} has missing transcript_title: {metadata}")
                items.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": metadata
                })
            logger.info(f"Checked transcript storage: found {len(items)} transcripts")
            for item in items:
                logger.debug(f"Transcript: {item}")
            return items
        except Exception as e:
            logger.error(f"Error checking transcript storage: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for single text or list of texts in batch"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings for texts: {texts[:100]}...")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {str(e)}")
            raise

    def fetch_transcripts(self, query: str = "", top_k: int = 500, date_range: Tuple[datetime, datetime] = None) -> List[Dict]:
        """Fetch transcripts with flexible filters and handle missing timestamps"""
        try:
            query_embedding = self.get_embeddings(query or "transcript")[0]
            filter_clause = {"user_id": str(self.user.id), "type": "transcript"}
            
            # Add date range filter if provided
            if date_range:
                start_date, end_date = date_range
                filter_clause["timestamp"] = {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_clause,
                include_metadata=True
            )
            
            logger.info(f"Transcript query returned {len(results.matches)} matches for query: {query[:100]}...")
            for match in results.matches:
                logger.debug(f"Match: id={match.id}, score={match.score}, metadata={match.metadata}")
            
            transcripts = []
            for match in results.matches:
                if match.score < self.retrieval_threshold:
                    continue
                
                metadata = match.metadata
                # Skip entries with missing or None timestamp
                if not metadata.get('timestamp'):
                    logger.warning(f"Skipping transcript {match.id} due to missing timestamp: {metadata}")
                    continue
                
                transcripts.append({
                    'id': match.id,
                    'text': metadata.get('text', ''),
                    'speaker': metadata.get('speaker', 'Unknown'),
                    'start_time': metadata.get('start_time', 0.0),
                    'end_time': metadata.get('end_time', 0.0),
                    'transcript_id': metadata.get('transcript_id', ''),
                    'transcript_title': metadata.get('transcript_title', 'Untitled'),
                    'transcript_date': metadata.get('transcript_date', ''),
                    'timestamp': metadata.get('timestamp'),
                    'type': 'transcript'
                })
            
            logger.info(f"Retrieved {len(transcripts)} valid transcripts")
            # Sort by timestamp, handling potential None values (should be filtered out)
            return sorted(transcripts, key=lambda x: x['timestamp'] or '1970-01-01T00:00:00Z', reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Error fetching transcripts: {str(e)}")
            raise RuntimeError(f"Failed to fetch transcripts: {str(e)}") from e

    def get_relevant_content(self, query: str, top_k: int = 10, date_range: Tuple[datetime, datetime] = None, 
                           content_type: str = None) -> Tuple[List[Dict], List[Dict]]:
        """Retrieve relevant content with increased top_k"""
        try:
            query_embedding = self.get_embeddings(query)[0]
            filter_clause = {"user_id": str(self.user.id)}
            
            # Add date range filter if provided
            if date_range:
                start_date, end_date = date_range
                filter_clause["timestamp"] = {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            
            # Add content type filter if specified
            if content_type in ["chat", "transcript"]:
                filter_clause["type"] = content_type
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k * 2,
                filter=filter_clause,
                include_metadata=True
            )
            
            logger.info(f"Pinecone query returned {len(results.matches)} matches for query: {query[:100]}...")
            for match in results.matches:
                logger.debug(f"Match: id={match.id}, score={match.score}, metadata={match.metadata}")
            
            relevant_chats = []
            relevant_transcripts = []
            
            for match in results.matches:
                if match.score < self.retrieval_threshold:
                    continue
                
                metadata = match.metadata
                if metadata.get('type') == 'chat':
                    relevant_chats.append({
                        'id': match.id,
                        'role': metadata.get('role', ''),
                        'content': metadata.get('content', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'type': 'chat'
                    })
                elif metadata.get('type') == 'transcript':
                    if not metadata.get('timestamp'):
                        logger.warning(f"Skipping transcript {match.id} due to missing timestamp: {metadata}")
                        continue
                    relevant_transcripts.append({
                        'id': match.id,
                        'text': metadata.get('text', ''),
                        'speaker': metadata.get('speaker', 'Unknown'),
                        'start_time': metadata.get('start_time', 0.0),
                        'end_time': metadata.get('end_time', 0.0),
                        'transcript_id': metadata.get('transcript_id', ''),
                        'transcript_title': metadata.get('transcript_title', 'Untitled'),
                        'transcript_date': metadata.get('transcript_date', ''),
                        'timestamp': metadata.get('timestamp'),
                        'type': 'transcript'
                    })
            
            logger.info(f"Filtered to {len(relevant_chats)} chats and {len(relevant_transcripts)} transcripts")
            return relevant_chats[:top_k], relevant_transcripts[:top_k]
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            raise RuntimeError(f"Failed to retrieve content: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def store_chat_message(self, message_id: str, role: str, content: str):
        """Store chat message in Pinecone and conversation memory with retry"""
        try:
            embedding = self.get_embeddings(content)[0]
            metadata = {
                'type': 'chat',
                'user_id': str(self.user.id),
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            # Validate metadata
            if not all(metadata.values()):
                logger.error(f"Invalid chat message metadata: {metadata}")
                raise ValueError("Chat message metadata contains None or empty values")
            
            response = self.pinecone_index.upsert(vectors=[
                {
                    'id': message_id,
                    'values': embedding,
                    'metadata': metadata
                }
            ])
            logger.info(f"Upserted chat message {message_id}: {response}")
            self.conversation_memory.append({"role": role, "content": content})
            self.conversation_memory = self.conversation_memory[-10:]
        except Exception as e:
            logger.error(f"Failed to store chat message: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def store_transcript_segment(self, segment_id: str, transcript_id: str, text: str, 
                              speaker: str, start_time: float, end_time: float):
        """Store transcript segment in Pinecone with Django ORM filtering and retry"""
        try:
            transcript = Transcript.objects.filter(
                Q(id=transcript_id) & Q(user=self.user)
            ).first()
            if not transcript:
                logger.error(f"Transcript {transcript_id} not found for user {self.user.id}")
                raise ValueError(f"Transcript {transcript_id} not found for user {self.user.id}")
            
            embedding = self.get_embeddings(text)[0]
            metadata = {
                'type': 'transcript',
                'user_id': str(self.user.id),
                'text': text,
                'speaker': speaker,
                'start_time': start_time,
                'end_time': end_time,
                'transcript_id': transcript_id,
                'transcript_title': transcript.title or 'Untitled',
                'transcript_date': transcript.created_at.isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            # Validate metadata
            if not all(metadata.values()):
                logger.error(f"Invalid transcript metadata: {metadata}")
                raise ValueError("Transcript metadata contains None or empty values")
            
            response = self.pinecone_index.upsert(vectors=[
                {
                    'id': segment_id,
                    'values': embedding,
                    'metadata': metadata
                }
            ])
            logger.info(f"Upserted transcript segment {segment_id}: {response}")
        except Exception as e:
            logger.error(f"Failed to store transcript segment: {str(e)}")
            raise

    def list_transcripts_by_date(self, date_str: str = None) -> List[Dict]:
        """List all transcripts, optionally for a specific date"""
        try:
            # Handle "all transcripts" or no date specified
            if not date_str or "all" in date_str.lower():
                transcripts = self.fetch_transcripts(top_k=500)
                logger.info(f"Retrieved {len(transcripts)} transcripts for 'all'")
                return transcripts
            
            # Parse date string (e.g., "yesterday" or "2025-05-05")
            if date_str.lower() == "yesterday":
                target_date = datetime.now().date() - timedelta(days=1)
            else:
                try:
                    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    logger.error(f"Invalid date format: {date_str}")
                    raise ValueError("Date must be 'yesterday' or in YYYY-MM-DD format")
            
            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = datetime.combine(target_date, datetime.max.time())
            
            transcripts = self.fetch_transcripts(
                query="",
                top_k=500,
                date_range=(start_date, end_date)
            )
            
            logger.info(f"Retrieved {len(transcripts)} transcripts for {date_str}")
            return transcripts
        except Exception as e:
            logger.error(f"Error listing transcripts for {date_str}: {str(e)}")
            raise RuntimeError(f"Failed to list transcripts: {str(e)}") from e

    def generate_response(self, user_message: str) -> str:
        """Generate response to user query using context-aware prompt"""
        try:
            # Handle transcript-related queries
            if "transcripts" in user_message.lower():
                date_match = re.search(r"(\d{4}-\d{2}-\d{2}|yesterday|all)", user_message, re.IGNORECASE)
                date_str = date_match.group(0) if date_match else None
                transcripts = self.list_transcripts_by_date(date_str)
                if not transcripts:
                    stored_transcripts = self.check_transcript_storage()
                    if not stored_transcripts:
                        return "No transcripts found. The data store may be empty or inaccessible. Please ensure transcripts are stored correctly."
                    return f"No transcripts found for {date_str or 'the specified criteria'}."
                
                # Summarize if too many transcripts
                max_display = 10
                if len(transcripts) > max_display:
                    response = f"Found {len(transcripts)} transcripts. Showing the {max_display} most recent:\n"
                else:
                    response = "Transcripts found:\n"
                
                for t in transcripts[:max_display]:
                    response += f"- {t['transcript_title']} (Speaker: {t['speaker']}, {t['transcript_date']}): {t['text'][:100]}...\n"
                
                if len(transcripts) > max_display:
                    response += f"\n...and {len(transcripts) - max_display} more. Please narrow your query for more details."
                return response

            return self._answer_general_question(user_message)
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"Sorry, I couldn't process your request due to an error: {str(e)}"

    def _answer_general_question(self, user_message: str) -> str:
        """Generate context-aware response with filtered context and increased token size"""
        try:
            user_chat_msg = ChatMessage.objects.create(
                user=self.user, role='user', content=user_message
            )
            self.store_chat_message(str(user_chat_msg.id), 'user', user_message)
            
            relevant_chats, relevant_transcripts = self.get_relevant_content(user_message)
            messages = [{"role": "system", "content": self.system_instruction}]
            
            # Add conversation memory as context
            messages.extend(
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.conversation_memory
            )
            
            # Add filtered transcript context
            messages.extend(
                {
                    "role": "system",
                    "content": (
                        f"Transcript Context from '{seg['transcript_title']}' "
                        f"(Speaker {seg['speaker']} at {seg['start_time']}s): {seg['text'][:500]}"
                    )
                }
                for seg in relevant_transcripts
            )
            
            # Add filtered chat history, avoiding duplicates
            messages.extend(
                {"role": chat['role'], "content": chat['content']}
                for chat in relevant_chats
                if {"role": chat['role'], "content": chat['content']} not in self.conversation_memory
            )
            
            # Fallback if no context is found
            if not relevant_chats and not relevant_transcripts:
                logger.warning(f"No relevant content found for query: {user_message[:100]}...")
                stored_transcripts = self.check_transcript_storage()
                if not stored_transcripts:
                    return "No chat history or transcripts found. The data store may be empty or inaccessible."
                messages.append({
                    "role": "system",
                    "content": (
                        "No relevant chat history or transcripts found. "
                        "Please provide more details or clarify your request."
                    )
                })
            
            # Add user query
            messages.append({"role": "user", "content": user_message})
            
            # Generate response with increased token size
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=4000,
                temperature=1.1
            )
            answer = response.choices[0].message.content.strip()
            
            # Store assistant response
            assistant_chat_msg = ChatMessage.objects.create(
                user=self.user, role='assistant', content=answer
            )
            self.store_chat_message(str(assistant_chat_msg.id), 'assistant', answer)
            
            return answer
        except Exception as e:
            logger.error(f"Question answering error: {str(e)}")
            return f"Failed to generate response: {str(e)}"
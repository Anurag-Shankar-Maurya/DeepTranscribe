import numpy as np
from openai import OpenAI
from django.conf import settings
from django.db.models import Q
from .models import ChatMessage, ChatMessageEmbedding
from core.models import Transcript, TranscriptEmbedding, TranscriptSegment
from collections import Counter
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, user):
        self.user = user
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_instruction = self._get_system_instruction()
        self.similarity_threshold = 0.7  # Higher threshold for better relevance
        self.max_context_segments = 10  # Limit context size for performance

    def _get_system_instruction(self):
        return """You are an intelligent assistant specialized in analyzing and answering questions about transcripts and conversations. 

Your capabilities include:
- Answering questions about specific transcript content
- Providing summaries and insights from transcripts
- Identifying speakers and their contributions
- Extracting key topics and themes
- Analyzing sentiment and tone
- Finding specific information across multiple sessions

Guidelines:
1. Always base your answers on the provided transcript context
2. If information isn't available in the transcripts, clearly state this
3. When referencing specific content, mention which transcript or speaker it came from
4. Be concise but comprehensive in your responses
5. Maintain user privacy - only access transcripts belonging to the current user
6. For general questions not related to transcripts, provide helpful general answers

Remember: You only have access to transcripts and chat history for the current user."""

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text using OpenAI's API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_relevant_transcript_content(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve relevant transcript segments for the current user only
        """
        query_embedding = self.get_embeddings(query)
        if not query_embedding:
            return []

        # Get transcript embeddings for current user only
        transcript_embeddings = TranscriptEmbedding.objects.filter(
            transcript__user=self.user,
            segment__isnull=False
        ).select_related('transcript', 'segment')

        relevant_segments = []
        
        for te in transcript_embeddings:
            if te.embedding and te.segment:
                similarity = self.cosine_similarity(query_embedding, te.embedding)
                
                if similarity >= self.similarity_threshold:
                    relevant_segments.append({
                        'segment_id': te.segment.id,
                        'text': te.segment.text,
                        'speaker': te.segment.speaker,
                        'start_time': te.segment.start_time,
                        'end_time': te.segment.end_time,
                        'confidence': te.segment.confidence,
                        'transcript_id': te.transcript.id,
                        'transcript_title': te.transcript.title,
                        'transcript_date': te.transcript.created_at,
                        'similarity': similarity
                    })

        # Sort by similarity and limit results
        relevant_segments.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_segments[:limit]

    def get_recent_chat_context(self, limit: int = 5) -> List[Dict]:
        """
        Get recent chat messages for context (user's messages only)
        """
        recent_messages = ChatMessage.objects.filter(
            user=self.user
        ).order_by('-timestamp')[:limit * 2]  # Get more to account for back-and-forth

        context = []
        for msg in reversed(recent_messages):
            context.append({
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            })
        
        return context

    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """
        Analyze the user's query to determine intent and extract parameters
        """
        query_lower = query.lower()
        
        # Check for specific commands/intents
        intent_patterns = {
            'list_transcripts': [
                r'list.*transcripts?',
                r'show.*transcripts?',
                r'what transcripts?',
                r'my transcripts?'
            ],
            'summarize': [
                r'summar[iy]ze?',
                r'summary',
                r'overview',
                r'key points?'
            ],
            'search_content': [
                r'find',
                r'search',
                r'look for',
                r'when.*said',
                r'who.*said'
            ],
            'speaker_analysis': [
                r'speaker',
                r'who.*talk',
                r'who.*speak',
                r'participants?'
            ],
            'time_based': [
                r'recent',
                r'last.*days?',
                r'yesterday',
                r'this week',
                r'today'
            ]
        }
        
        detected_intent = 'general'
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intent = intent
                    break
            if detected_intent != 'general':
                break
        
        # Extract time references
        time_match = re.search(r'last (\d+) (days?|weeks?|months?)', query_lower)
        time_filter = None
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2).rstrip('s')
            if unit == 'day':
                time_filter = datetime.now() - timedelta(days=num)
            elif unit == 'week':
                time_filter = datetime.now() - timedelta(weeks=num)
            elif unit == 'month':
                time_filter = datetime.now() - timedelta(days=num*30)
        
        return {
            'intent': detected_intent,
            'time_filter': time_filter,
            'original_query': query
        }

    def handle_list_transcripts(self, intent_data: Dict) -> str:
        """Handle requests to list transcripts"""
        query_filter = Q(user=self.user)
        
        if intent_data['time_filter']:
            query_filter &= Q(created_at__gte=intent_data['time_filter'])
        
        transcripts = Transcript.objects.filter(query_filter).order_by('-created_at')[:20]
        
        if not transcripts.exists():
            return "You don't have any transcripts yet. Start by creating a new transcript session."
        
        response = "Here are your transcripts:\n\n"
        for i, transcript in enumerate(transcripts, 1):
            status = "✅ Complete" if transcript.is_complete else "🔄 In Progress"
            duration = f"{transcript.duration}s" if transcript.duration else "Unknown"
            response += f"{i}. **{transcript.title}**\n"
            response += f"   📅 {transcript.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            response += f"   ⏱️ Duration: {duration}\n"
            response += f"   📊 Status: {status}\n\n"
        
        return response

    def handle_summarize_request(self, intent_data: Dict) -> str:
        """Handle summarization requests"""
        # Get recent transcripts for summarization
        query_filter = Q(user=self.user)
        if intent_data['time_filter']:
            query_filter &= Q(created_at__gte=intent_data['time_filter'])
        
        transcripts = Transcript.objects.filter(query_filter).order_by('-created_at')[:5]
        
        if not transcripts.exists():
            return "No transcripts found to summarize."
        
        # Collect transcript content
        transcript_content = []
        for transcript in transcripts:
            segments = transcript.segments.all().order_by('start_time')[:20]  # Limit segments
            content = f"Transcript: {transcript.title} ({transcript.created_at.strftime('%Y-%m-%d')})\n"
            content += "\n".join([f"Speaker {seg.speaker}: {seg.text}" for seg in segments])
            transcript_content.append(content)
        
        combined_content = "\n\n---\n\n".join(transcript_content)
        
        # Generate summary using OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of transcript content. Focus on key topics, decisions, and important points discussed."},
                    {"role": "user", "content": f"Please provide a comprehensive summary of these transcript sessions:\n\n{combined_content}"}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "I encountered an error while generating the summary. Please try again."

    def handle_speaker_analysis(self, intent_data: Dict) -> str:
        """Handle speaker-related queries"""
        query_filter = Q(transcript__user=self.user)
        if intent_data['time_filter']:
            query_filter &= Q(transcript__created_at__gte=intent_data['time_filter'])
        
        segments = TranscriptSegment.objects.filter(query_filter)
        
        if not segments.exists():
            return "No transcript segments found for speaker analysis."
        
        # Analyze speaker activity
        speaker_stats = {}
        total_segments = 0
        
        for segment in segments:
            speaker_id = segment.speaker if segment.speaker is not None else "Unknown"
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    'segments': 0,
                    'total_words': 0,
                    'total_time': 0
                }
            
            speaker_stats[speaker_id]['segments'] += 1
            speaker_stats[speaker_id]['total_words'] += len(segment.text.split())
            speaker_stats[speaker_id]['total_time'] += (segment.end_time - segment.start_time)
            total_segments += 1
        
        # Format response
        response = "Speaker Analysis:\n\n"
        for speaker_id, stats in sorted(speaker_stats.items(), key=lambda x: x[1]['segments'], reverse=True):
            percentage = (stats['segments'] / total_segments) * 100
            avg_words = stats['total_words'] / stats['segments'] if stats['segments'] > 0 else 0
            
            response += f"**Speaker {speaker_id}:**\n"
            response += f"  • Segments: {stats['segments']} ({percentage:.1f}%)\n"
            response += f"  • Total words: {stats['total_words']}\n"
            response += f"  • Average words per segment: {avg_words:.1f}\n"
            response += f"  • Total speaking time: {stats['total_time']:.1f}s\n\n"
        
        return response

    def generate_response(self, user_message: str) -> str:
        """Main method to generate response to user query"""
        try:
            # Store user message
            user_chat_msg = ChatMessage.objects.create(
                user=self.user,
                role='user',
                content=user_message
            )
            
            # Generate and store embedding for user message
            user_embedding = self.get_embeddings(user_message)
            if user_embedding:
                ChatMessageEmbedding.objects.create(
                    chat_message=user_chat_msg,
                    embedding=user_embedding
                )
            
            # Analyze query intent
            intent_data = self.analyze_query_intent(user_message)
            
            # Handle specific intents
            if intent_data['intent'] == 'list_transcripts':
                response = self.handle_list_transcripts(intent_data)
            elif intent_data['intent'] == 'summarize':
                response = self.handle_summarize_request(intent_data)
            elif intent_data['intent'] == 'speaker_analysis':
                response = self.handle_speaker_analysis(intent_data)
            else:
                # Handle general queries with transcript context
                response = self._handle_contextual_query(user_message, intent_data)
            
            # Store assistant response
            assistant_chat_msg = ChatMessage.objects.create(
                user=self.user,
                role='assistant',
                content=response
            )
            
            # Generate and store embedding for assistant response
            assistant_embedding = self.get_embeddings(response)
            if assistant_embedding:
                ChatMessageEmbedding.objects.create(
                    chat_message=assistant_chat_msg,
                    embedding=assistant_embedding
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while processing your request. Please try again."

    def _handle_contextual_query(self, user_message: str, intent_data: Dict) -> str:
        """Handle general queries with transcript context"""
        # Get relevant transcript content
        relevant_segments = self.get_relevant_transcript_content(user_message, self.max_context_segments)
        
        # Get recent chat context
        chat_context = self.get_recent_chat_context(5)
        
        # Prepare messages for OpenAI
        messages = [{"role": "system", "content": self.system_instruction}]
        
        # Add chat context
        for chat_msg in chat_context[:-1]:  # Exclude the current message
            messages.append({"role": chat_msg['role'], "content": chat_msg['content']})
        
        # Add relevant transcript context
        if relevant_segments:
            context_text = "Relevant transcript content:\n\n"
            for segment in relevant_segments:
                context_text += f"From '{segment['transcript_title']}' ({segment['transcript_date'].strftime('%Y-%m-%d')}):\n"
                context_text += f"Speaker {segment['speaker']}: {segment['text']}\n"
                context_text += f"Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s\n\n"
            
            messages.append({"role": "system", "content": context_text})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # If no relevant transcript content was found, mention it
            if not relevant_segments and any(keyword in user_message.lower() for keyword in ['transcript', 'said', 'mentioned', 'discussed']):
                answer += "\n\n*Note: I couldn't find relevant transcript content for your query. You may want to check if you have transcripts that contain this information.*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "I'm having trouble processing your request right now. Please try again in a moment."

    def get_user_transcript_stats(self) -> Dict:
        """Get statistics about user's transcripts"""
        transcripts = Transcript.objects.filter(user=self.user)
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        
        return {
            'total_transcripts': transcripts.count(),
            'completed_transcripts': transcripts.filter(is_complete=True).count(),
            'total_segments': segments.count(),
            'total_duration': sum(t.duration for t in transcripts if t.duration),
            'unique_speakers': segments.values('speaker').distinct().count()
        }
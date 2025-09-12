import numpy as np
import google.generativeai as genai
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
<<<<<<< HEAD
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_instruction = self._get_system_instruction()
        self.similarity_threshold = 0.6  # Balanced threshold
        self.max_context_segments = 15  # More context for better responses

    def _get_system_instruction(self):
        return """You are an intelligent assistant specialized in helping users analyze and understand their personal transcripts and conversations.

Your capabilities include:
- Listing and counting user's transcripts
- Providing summaries and insights from transcripts
- Identifying speakers and their contributions
- Extracting key topics and themes
- Finding specific information across transcript sessions
- Analyzing conversation patterns and statistics

Guidelines:
1. Always base your answers on the provided transcript data for this specific user
2. When listing transcripts, provide clear details like title, date, duration, and status
3. For content searches, reference which transcript the information came from
4. Be conversational and helpful in your responses
5. If no relevant transcript data is found, clearly explain this to the user
6. Provide specific numbers, dates, and details when available
7. Use bullet points and clear formatting for better readability

Remember: You have access to this user's transcript data and should provide specific, helpful responses based on their actual data."""

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
=======
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
>>>>>>> 7a4076b101b0a58522b91f812aa17c56430bbdf6

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
<<<<<<< HEAD
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
=======
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
>>>>>>> 7a4076b101b0a58522b91f812aa17c56430bbdf6
            
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_user_transcripts(self, limit: int = None) -> List[Dict]:
        """Get user's transcripts with metadata"""
        query = Transcript.objects.filter(user=self.user).order_by('-created_at')
        if limit:
            query = query[:limit]
        
<<<<<<< HEAD
        transcripts = []
        for transcript in query:
            transcripts.append({
                'id': transcript.id,
                'title': transcript.title,
                'created_at': transcript.created_at,
                'duration': transcript.duration,
                'is_complete': transcript.is_complete,
                'segment_count': transcript.segments.count()
            })
        
        return transcripts

    def get_relevant_transcript_content(self, query: str, limit: int = 15) -> List[Dict]:
        """Retrieve relevant transcript segments for the current user only"""
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

    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """Analyze the user's query to determine intent and extract parameters"""
        query_lower = query.lower()
        
        # Check for specific commands/intents
        intent_patterns = {
            'count_transcripts': [
                r'how many.*transcripts?',
                r'count.*transcripts?',
                r'number.*transcripts?',
                r'how many.*sessions?',
                r'count.*sessions?'
            ],
            'list_transcripts': [
                r'list.*transcripts?',
                r'show.*transcripts?',
                r'what transcripts?',
                r'my transcripts?',
                r'latest.*transcripts?',
                r'recent.*transcripts?',
                r'give me.*transcripts?'
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
        
        # Extract numbers for "latest X transcripts"
        number_match = re.search(r'latest (\d+)|(\d+) transcripts?|(\d+) sessions?', query_lower)
        limit = None
        if number_match:
            limit = int(number_match.group(1) or number_match.group(2) or number_match.group(3))
        
        return {
            'intent': detected_intent,
            'limit': limit,
            'original_query': query
        }

    def handle_count_transcripts(self) -> str:
        """Handle requests to count transcripts"""
        total_count = Transcript.objects.filter(user=self.user).count()
        completed_count = Transcript.objects.filter(user=self.user, is_complete=True).count()
        in_progress_count = total_count - completed_count
        
        response = f"📊 **Your Transcript Statistics:**\n\n"
        response += f"• **Total transcripts:** {total_count}\n"
        response += f"• **Completed:** {completed_count}\n"
        response += f"• **In progress:** {in_progress_count}\n"
        
        if total_count > 0:
            # Get latest transcript info
            latest = Transcript.objects.filter(user=self.user).order_by('-created_at').first()
            response += f"\n**Latest transcript:** {latest.title} ({latest.created_at.strftime('%Y-%m-%d %H:%M')})"
        
        return response

    def handle_list_transcripts(self, limit: int = None) -> str:
        """Handle requests to list transcripts"""
        transcripts = self.get_user_transcripts(limit)
        
        if not transcripts:
            return "📝 You don't have any transcripts yet. Start by creating a new transcript session!"
        
        response = f"📋 **Your Transcripts"
        if limit:
            response += f" (Latest {min(limit, len(transcripts))})"
        response += ":**\n\n"
        
        for i, transcript in enumerate(transcripts, 1):
            status = "✅ Complete" if transcript['is_complete'] else "🔄 In Progress"
            duration = f"{transcript['duration']}s" if transcript['duration'] else "Unknown"
            
            response += f"**{i}. {transcript['title']}**\n"
            response += f"   📅 {transcript['created_at'].strftime('%Y-%m-%d %H:%M')}\n"
            response += f"   ⏱️ Duration: {duration}\n"
            response += f"   📊 Status: {status}\n"
            response += f"   💬 Segments: {transcript['segment_count']}\n\n"
        
        return response

    def handle_summarize_request(self, limit: int = 5) -> str:
        """Handle summarization requests"""
        transcripts = self.get_user_transcripts(limit)
        
        if not transcripts:
            return "📝 No transcripts found to summarize."
        
        # Collect transcript content
        transcript_content = []
        for transcript_info in transcripts:
            transcript = Transcript.objects.get(id=transcript_info['id'])
            segments = transcript.segments.all().order_by('start_time')[:20]  # Limit segments
            if segments:
                content = f"Transcript: {transcript.title} ({transcript.created_at.strftime('%Y-%m-%d')})\n"
                content += "\n".join([f"Speaker {seg.speaker}: {seg.text}" for seg in segments])
                transcript_content.append(content)
        
        if not transcript_content:
            return "📝 No transcript content available to summarize."
        
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
            return "📋 **Summary of Your Recent Transcripts:**\n\n" + response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "❌ I encountered an error while generating the summary. Please try again."

    def handle_speaker_analysis(self) -> str:
        """Handle speaker-related queries"""
        segments = TranscriptSegment.objects.filter(transcript__user=self.user)
        
        if not segments.exists():
            return "📝 No transcript segments found for speaker analysis."
        
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
        response = "🎤 **Speaker Analysis:**\n\n"
        for speaker_id, stats in sorted(speaker_stats.items(), key=lambda x: x[1]['segments'], reverse=True):
            percentage = (stats['segments'] / total_segments) * 100
            avg_words = stats['total_words'] / stats['segments'] if stats['segments'] > 0 else 0
            
            response += f"**Speaker {speaker_id}:**\n"
            response += f"  • Segments: {stats['segments']} ({percentage:.1f}%)\n"
            response += f"  • Total words: {stats['total_words']}\n"
            response += f"  • Average words per segment: {avg_words:.1f}\n"
            response += f"  • Total speaking time: {stats['total_time']:.1f}s\n\n"
        
        return response
=======
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
>>>>>>> 7a4076b101b0a58522b91f812aa17c56430bbdf6

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
            if intent_data['intent'] == 'count_transcripts':
                response = self.handle_count_transcripts()
            elif intent_data['intent'] == 'list_transcripts':
                response = self.handle_list_transcripts(intent_data.get('limit'))
            elif intent_data['intent'] == 'summarize':
                response = self.handle_summarize_request()
            elif intent_data['intent'] == 'speaker_analysis':
                response = self.handle_speaker_analysis()
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
            logger.error(f"Error generating response for user {self.user.id}: {e}")
            return "❌ I encountered an error while processing your request. Please try again."

<<<<<<< HEAD
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
            context_text = "Here are the relevant transcript segments for this user:\n\n"
            for segment in relevant_segments:
                context_text += f"From '{segment['transcript_title']}' ({segment['transcript_date'].strftime('%Y-%m-%d')}):\n"
                context_text += f"Speaker {segment['speaker']}: {segment['text']}\n"
                context_text += f"Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s\n\n"
            
            messages.append({"role": "system", "content": context_text})
        else:
            # Add user stats if no relevant content found
            stats = self.get_user_transcript_stats()
            if stats['total_transcripts'] > 0:
                stats_text = f"User has {stats['total_transcripts']} transcripts with {stats['total_segments']} total segments."
                messages.append({"role": "system", "content": stats_text})
        
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
            
            # If no relevant transcript content was found, provide helpful guidance
            if not relevant_segments and any(keyword in user_message.lower() for keyword in ['transcript', 'said', 'mentioned', 'discussed']):
                stats = self.get_user_transcript_stats()
                if stats['total_transcripts'] == 0:
                    answer += "\n\n💡 *You don't have any transcripts yet. Start by creating a new transcript session to begin using the AI assistant features!*"
                else:
                    answer += f"\n\n💡 *I couldn't find relevant content in your {stats['total_transcripts']} transcripts for this specific query. Try asking about general topics or use commands like 'list my transcripts' or 'summarize my conversations'.*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "❌ I'm having trouble processing your request right now. Please try again in a moment."

    def get_recent_chat_context(self, limit: int = 5) -> List[Dict]:
        """Get recent chat messages for context (user's messages only)"""
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
=======
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
>>>>>>> 7a4076b101b0a58522b91f812aa17c56430bbdf6

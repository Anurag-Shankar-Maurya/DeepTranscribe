"""
Memory management system for chatbot with multiple memory types:
- Buffer Memory: Stores recent conversation history
- Summary Memory: Maintains a running summary of conversations
- Summary Buffer Memory: Combination of summary + recent messages
- Graph Memory: Knowledge graph of entities and relationships
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import json
import re
from django.conf import settings
import google.generativeai as genai
from .models import ChatMessage


class MemoryType:
    BUFFER = "buffer"
    SUMMARY = "summary"
    SUMMARY_BUFFER = "summary_buffer"
    GRAPH = "graph"


class BufferMemory:
    """Stores recent k messages in conversation history"""
    
    def __init__(self, user, k=10):
        self.user = user
        self.k = k
    
    def get_memory(self) -> List[Dict]:
        """Get last k messages"""
        messages = ChatMessage.objects.filter(
            user=self.user
        ).order_by('-timestamp')[:self.k]
        
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            }
            for msg in reversed(messages)
        ]
    
    def get_context_string(self) -> str:
        """Get memory as formatted string"""
        messages = self.get_memory()
        if not messages:
            return ""
        
        context = "Recent conversation history:\n"
        for msg in messages:
            context += f"{msg['role']}: {msg['content']}\n"
        return context


class SummaryMemory:
    """Maintains a running summary of the conversation"""
    
    def __init__(self, user):
        self.user = user
        genai.configure(api_key=settings.GEMINI_API_KEY)
    
    def generate_summary(self, messages: List[Dict]) -> str:
        """Generate summary from messages"""
        if not messages:
            return ""
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        prompt = f"""Provide a concise summary of the following conversation, 
        highlighting key topics, decisions, and important information:
        
        {conversation_text}
        
        Summary:"""
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed."
    
    def get_memory(self) -> str:
        """Get conversation summary"""
        # Get all messages from the last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        messages = ChatMessage.objects.filter(
            user=self.user,
            timestamp__gte=week_ago
        ).order_by('timestamp')
        
        message_list = [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            }
            for msg in messages
        ]
        
        return self.generate_summary(message_list)
    
    def get_context_string(self) -> str:
        """Get memory as formatted string"""
        summary = self.get_memory()
        if not summary:
            return ""
        return f"Conversation summary:\n{summary}\n"


class SummaryBufferMemory:
    """Combines summary of old messages with recent buffer"""
    
    def __init__(self, user, buffer_size=5, summary_threshold=20):
        self.user = user
        self.buffer_size = buffer_size
        self.summary_threshold = summary_threshold
        self.buffer_memory = BufferMemory(user, k=buffer_size)
        self.summary_memory = SummaryMemory(user)
    
    def get_memory(self) -> Dict:
        """Get combined summary and recent messages"""
        # Get recent messages
        recent_messages = self.buffer_memory.get_memory()
        
        # Get older messages for summary (beyond recent buffer)
        week_ago = datetime.now() - timedelta(days=7)
        older_messages = ChatMessage.objects.filter(
            user=self.user,
            timestamp__gte=week_ago
        ).order_by('timestamp')[:-self.buffer_size] if ChatMessage.objects.filter(
            user=self.user,
            timestamp__gte=week_ago
        ).count() > self.buffer_size else []
        
        summary = ""
        if older_messages:
            older_msg_list = [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp
                }
                for msg in older_messages
            ]
            summary = self.summary_memory.generate_summary(older_msg_list)
        
        return {
            'summary': summary,
            'recent_messages': recent_messages
        }
    
    def get_context_string(self) -> str:
        """Get memory as formatted string"""
        memory = self.get_memory()
        context = ""
        
        if memory['summary']:
            context += f"Previous conversation summary:\n{memory['summary']}\n\n"
        
        if memory['recent_messages']:
            context += "Recent messages:\n"
            for msg in memory['recent_messages']:
                context += f"{msg['role']}: {msg['content']}\n"
        
        return context


class GraphMemory:
    """Knowledge graph memory that tracks entities and relationships"""
    
    def __init__(self, user):
        self.user = user
        genai.configure(api_key=settings.GEMINI_API_KEY)
    
    def extract_entities_and_relations(self, text: str) -> Dict:
        """Extract entities and relationships from text"""
        prompt = f"""Extract entities (people, places, organizations, concepts) and their relationships from the following text.
        Return the result as a JSON object with 'entities' (list of strings) and 'relationships' (list of objects with 'subject', 'relation', 'object').
        
        Text: {text}
        
        JSON:"""
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            
            # Try to parse JSON from response
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\n?', '', response_text)
            response_text = re.sub(r'```\n?', '', response_text)
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {'entities': [], 'relationships': []}
    
    def build_knowledge_graph(self) -> Dict:
        """Build knowledge graph from conversation history"""
        # Get messages from last 30 days
        month_ago = datetime.now() - timedelta(days=30)
        messages = ChatMessage.objects.filter(
            user=self.user,
            timestamp__gte=month_ago
        ).order_by('timestamp')
        
        entities = set()
        relationships = []
        
        for msg in messages:
            if len(msg.content) > 20:  # Only process substantive messages
                extracted = self.extract_entities_and_relations(msg.content)
                entities.update(extracted.get('entities', []))
                relationships.extend(extracted.get('relationships', []))
        
        return {
            'entities': list(entities),
            'relationships': relationships
        }
    
    def get_memory(self) -> Dict:
        """Get knowledge graph"""
        return self.build_knowledge_graph()
    
    def get_context_string(self) -> str:
        """Get memory as formatted string"""
        graph = self.get_memory()
        
        if not graph['entities'] and not graph['relationships']:
            return ""
        
        context = "Knowledge graph from past conversations:\n"
        
        if graph['entities']:
            context += f"Entities mentioned: {', '.join(graph['entities'][:20])}\n"
        
        if graph['relationships']:
            context += "Key relationships:\n"
            for rel in graph['relationships'][:10]:
                context += f"- {rel.get('subject', '')} {rel.get('relation', '')} {rel.get('object', '')}\n"
        
        return context


class MemoryManager:
    """Main memory manager that handles all memory types"""
    
    def __init__(self, user, memory_type=MemoryType.SUMMARY_BUFFER):
        self.user = user
        self.memory_type = memory_type
        self._memory = None
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize the selected memory type"""
        if self.memory_type == MemoryType.BUFFER:
            self._memory = BufferMemory(self.user)
        elif self.memory_type == MemoryType.SUMMARY:
            self._memory = SummaryMemory(self.user)
        elif self.memory_type == MemoryType.SUMMARY_BUFFER:
            self._memory = SummaryBufferMemory(self.user)
        elif self.memory_type == MemoryType.GRAPH:
            self._memory = GraphMemory(self.user)
        else:
            self._memory = SummaryBufferMemory(self.user)  # Default
    
    def get_context(self) -> str:
        """Get memory context as string"""
        if self._memory:
            return self._memory.get_context_string()
        return ""
    
    def get_memory_data(self) -> Dict:
        """Get raw memory data"""
        if self._memory:
            return self._memory.get_memory()
        return {}
    
    def set_memory_type(self, memory_type: str):
        """Change memory type"""
        if memory_type in [MemoryType.BUFFER, MemoryType.SUMMARY, 
                          MemoryType.SUMMARY_BUFFER, MemoryType.GRAPH]:
            self.memory_type = memory_type
            self._initialize_memory()

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.conf import settings

# For advanced search (requires PostgreSQL)
try:
    from django.contrib.postgres.search import SearchVectorField
    from django.contrib.postgres.indexes import GinIndex
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

User = get_user_model()

class ChatSession(models.Model):
    """
    Groups chat messages into a specific conversation thread.
    Stores the 'memory' summary for that specific thread.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    session_id = models.CharField(max_length=255, unique=True)
    summary = models.TextField(blank=True, default="") # The persistent memory
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Session {self.session_id} - {self.user.username}"

class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    
    # Link message to a specific session
    session = models.ForeignKey(
        ChatSession, 
        on_delete=models.CASCADE, 
        related_name='messages',
        null=True, # Allow null temporarily for migration of old messages
        blank=True
    )
    
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username} - {self.role} - {self.content[:50]}"

class ChatMessageEmbedding(models.Model):
    chat_message = models.OneToOneField(ChatMessage, on_delete=models.CASCADE, related_name='embedding')
    embedding = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Embedding for message {self.chat_message.id}"

class Entity(models.Model):
    """
    Stores specific nouns: 'Sarah', 'Project X', 'Python'.
    Used for 'Precise' retrieval and Graph linking.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    entity_type = models.CharField(max_length=50) # e.g., PERSON, LOCATION, CODE_SNIPPET
    description = models.TextField(blank=True) # AI generated description
    last_mentioned = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'name')

    def __str__(self):
        return f"{self.name} ({self.entity_type})"

class KnowledgeTriple(models.Model):
    """
    The Knowledge Graph edges: (Subject) -> [Predicate] -> (Object)
    e.g., (User) -> [LIKES] -> (Stoicism)
    e.g., (Project X) -> [HAS_DEADLINE] -> (Jan 10th)
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    subject = models.ForeignKey(Entity, related_name='relations_as_subject', on_delete=models.CASCADE)
    predicate = models.CharField(max_length=100) # e.g., "is_located_in", "assigned_to"
    object_entity = models.ForeignKey(Entity, related_name='relations_as_object', on_delete=models.CASCADE)
    source_message = models.ForeignKey('ChatMessage', null=True, on_delete=models.SET_NULL)
    weight = models.FloatField(default=1.0) # How strong/recent is this connection?

    def __str__(self):
        return f"{self.subject.name} -> {self.predicate} -> {self.object_entity.name}"

class HierarchicalSummary(models.Model):
    """
    Macro Memory: Summaries of time buckets or topic clusters.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    level = models.CharField(max_length=20, choices=[('day', 'Day'), ('week', 'Week'), ('month', 'Month')])
    content = models.TextField()
    embedding = models.JSONField(null=True) # Vector of the summary itself

    def __str__(self):
        return f"{self.level} summary: {self.start_date.date()} - {self.end_date.date()}"

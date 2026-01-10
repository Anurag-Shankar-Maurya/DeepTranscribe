from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

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

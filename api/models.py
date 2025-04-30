from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
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

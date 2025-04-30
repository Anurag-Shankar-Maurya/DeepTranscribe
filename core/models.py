from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class Transcript(models.Model):
    """Model to store transcript information"""
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transcripts')
    is_complete = models.BooleanField(default=False)
    duration = models.IntegerField(default=0)  # Duration in seconds
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created_at']


class TranscriptSegment(models.Model):
    """Model to store individual transcript segments"""
    transcript = models.ForeignKey(Transcript, on_delete=models.CASCADE, related_name='segments')
    text = models.TextField()
    speaker = models.IntegerField(null=True, blank=True)
    start_time = models.FloatField(default=0.0)  # Start time in seconds
    end_time = models.FloatField(default=0.0)  # End time in seconds
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Speaker {self.speaker}: {self.text[:50]}..."
    
    class Meta:
        ordering = ['start_time']


class TranscriptSettings(models.Model):
    """Model to store transcript settings"""
    transcript = models.OneToOneField(Transcript, on_delete=models.CASCADE, related_name='settings')
    model = models.CharField(max_length=50, default='nova-3')
    language = models.CharField(max_length=10, default='multi')
    diarize = models.BooleanField(default=True)
    punctuate = models.BooleanField(default=True)
    numerals = models.BooleanField(default=True)
    smart_format = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Settings for {self.transcript.title}"


class TranscriptEmbedding(models.Model):
    """Model to store vector embeddings for transcripts or segments"""
    transcript = models.ForeignKey(Transcript, on_delete=models.CASCADE, related_name='embeddings', null=True, blank=True)
    segment = models.ForeignKey(TranscriptSegment, on_delete=models.CASCADE, related_name='embeddings', null=True, blank=True)
    embedding = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if self.segment:
            return f"Embedding for segment {self.segment.id}"
        elif self.transcript:
            return f"Embedding for transcript {self.transcript.id}"
        else:
            return "Embedding with no linked transcript or segment"

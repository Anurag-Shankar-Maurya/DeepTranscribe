from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Transcript, TranscriptSegment, TranscriptEmbedding
from api.utils import generate_embedding

@receiver(post_save, sender=Transcript)
def create_transcript_embedding(sender, instance, created, **kwargs):
    if created:
        embedding = generate_embedding(instance.title)
        TranscriptEmbedding.objects.create(transcript=instance, embedding=embedding)

@receiver(post_save, sender=TranscriptSegment)
def create_segment_embedding(sender, instance, created, **kwargs):
    if created:
        embedding = generate_embedding(instance.text)
        TranscriptEmbedding.objects.create(segment=instance, embedding=embedding)

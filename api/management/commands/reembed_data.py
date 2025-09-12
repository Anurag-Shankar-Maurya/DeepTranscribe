
from django.core.management.base import BaseCommand
from api.models import ChatMessage, ChatMessageEmbedding
from core.models import Transcript, TranscriptSegment, TranscriptEmbedding
from api.utils import generate_embedding

class Command(BaseCommand):
    help = 'Re-generates embeddings for all transcripts, segments, and clears chat history.'

    def handle(self, *args, **options):
        self.stdout.write("Starting embedding regeneration process...")

        # 1. Clear old embeddings and chat messages
        self.stdout.write("Deleting old TranscriptEmbeddings...")
        TranscriptEmbedding.objects.all().delete()
        self.stdout.write("Deleting old ChatMessageEmbeddings...")
        ChatMessageEmbedding.objects.all().delete()
        self.stdout.write("Deleting old ChatMessages...")
        ChatMessage.objects.all().delete()

        # 2. Re-generate embeddings for Transcripts
        self.stdout.write("Re-generating embeddings for Transcripts...")
        for transcript in Transcript.objects.all():
            embedding = generate_embedding(transcript.title)
            if embedding:
                TranscriptEmbedding.objects.create(transcript=transcript, embedding=embedding)
        self.stdout.write(self.style.SUCCESS(f"Processed {Transcript.objects.count()} transcripts."))

        # 3. Re-generate embeddings for TranscriptSegments
        self.stdout.write("Re-generating embeddings for TranscriptSegments...")
        for segment in TranscriptSegment.objects.all():
            embedding = generate_embedding(segment.text)
            if embedding:
                # Note: The original signal created a TranscriptEmbedding, so we do the same.
                # This might need adjustment if the model expects segment-specific embeddings.
                TranscriptEmbedding.objects.create(segment=segment, embedding=embedding)
        self.stdout.write(self.style.SUCCESS(f"Processed {TranscriptSegment.objects.count()} transcript segments."))

        self.stdout.write(self.style.SUCCESS("Embedding regeneration complete!"))


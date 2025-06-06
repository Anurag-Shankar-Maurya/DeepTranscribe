from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.conf import settings
from core.models import Transcript, TranscriptSegment
from .serializers import TranscriptSerializer, TranscriptSegmentSerializer
from openai import OpenAI
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from openai import OpenAI
from core.models import Transcript
from .models import ChatMessage, ChatMessageEmbedding
from .chatbot_service import ChatbotService
import logging

logger = logging.getLogger(__name__)

class TranscriptViewSet(viewsets.ModelViewSet):
    """ViewSet for viewing and editing transcripts."""
    serializer_class = TranscriptSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return transcripts for the current user."""
        return Transcript.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Set the user when creating a transcript."""
        serializer.save(user=self.request.user)


class TranscriptSegmentViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing transcript segments."""
    serializer_class = TranscriptSegmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return segments for the current user's transcripts."""
        transcript_id = self.request.query_params.get('transcript_id')
        if transcript_id:
            return TranscriptSegment.objects.filter(
                transcript_id=transcript_id,
                transcript__user=self.request.user
            )
        return TranscriptSegment.objects.filter(transcript__user=self.request.user)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def app_settings(request):
    """Return application settings."""
    return Response({
        'deepgram': {
            'models': ['nova-3', 'enhanced', 'base'],
            'languages': [
                {'code': 'multi', 'name': 'Multiple Languages'},
                {'code': 'en', 'name': 'English'},
                {'code': 'en-US', 'name': 'English (US)'},
                {'code': 'en-GB', 'name': 'English (UK)'},
                {'code': 'en-AU', 'name': 'English (Australia)'},
                {'code': 'en-IN', 'name': 'English (India)'},
                {'code': 'fr', 'name': 'French'},
                {'code': 'de', 'name': 'German'},
                {'code': 'es', 'name': 'Spanish'},
                {'code': 'it', 'name': 'Italian'},
                {'code': 'ja', 'name': 'Japanese'},
                {'code': 'ko', 'name': 'Korean'},
                {'code': 'pt', 'name': 'Portuguese'},
                {'code': 'ru', 'name': 'Russian'},
                {'code': 'zh', 'name': 'Chinese'},
                {'code': 'hi', 'name': 'Hindi'},
                {'code': 'ar', 'name': 'Arabic'},
            ],
        }
    })


class ChatbotAPIView(APIView):
    """API view to handle chatbot requests with improved security and performance."""
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request, *args, **kwargs):
        user = request.user
        user_message = request.data.get('message', '').strip()
        
        if not user_message:
            return Response(
                {'error': 'Message is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate message length
        if len(user_message) > 2000:
            return Response(
                {'error': 'Message too long. Please limit to 2000 characters.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Initialize chatbot service for the specific user
            chatbot_service = ChatbotService(user)
            
            # Generate response
            answer = chatbot_service.generate_response(user_message)
            
            # Get user stats for additional context
            user_stats = chatbot_service.get_user_transcript_stats()
            
            return Response({
                'response': answer,
                'user_stats': user_stats,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Chatbot error for user {user.id}: {str(e)}")
            return Response(
                {'error': 'An error occurred while processing your request. Please try again.'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
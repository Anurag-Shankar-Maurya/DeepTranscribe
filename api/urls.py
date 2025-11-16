from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, analytics_views

router = DefaultRouter()
router.register(r'transcripts', views.TranscriptViewSet, basename='transcript')
router.register(r'segments', views.TranscriptSegmentViewSet, basename='segment')

app_name = 'api'

urlpatterns = [
    path('', include(router.urls)),
    path('settings/', views.app_settings, name='settings'),
    path('chatbot/', views.ChatbotAPIView.as_view(), name='chatbot'),
    
    # Enhanced analytics views
    path('dashboard/', analytics_views.user_dashboard, name='user_dashboard'),
    path('transcripts/<int:pk>/enhanced/', analytics_views.transcript_detail_enhanced, name='transcript_detail_enhanced'),
    path('transcripts/list/', analytics_views.user_transcripts_api, name='user_transcripts_api'),
    path('transcripts/<int:pk>/analytics/', analytics_views.transcript_analytics_api, name='transcript_analytics_api'),
]

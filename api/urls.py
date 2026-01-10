from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'transcripts', views.TranscriptViewSet, basename='transcript')
router.register(r'segments', views.TranscriptSegmentViewSet, basename='segment')

app_name = 'api'

urlpatterns = [
    path('', include(router.urls)),
    path('settings/', views.app_settings, name='settings'),
    path('chatbot/', views.ChatbotView.as_view(), name='chatbot'),
]

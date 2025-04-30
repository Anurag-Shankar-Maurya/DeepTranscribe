from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.index, name='index'),
    path('transcribe/', views.transcribe, name='transcribe'),
    path('transcripts/', views.transcript_list, name='transcript_list'),
    path('transcripts/<int:pk>/', views.transcript_detail, name='transcript_detail'),
    path('transcripts/<int:pk>/edit/', views.transcript_edit, name='transcript_edit'),
    path('transcripts/<int:pk>/delete/', views.transcript_delete, name='transcript_delete'),
    path('transcripts/<int:pk>/download/<str:format>/', views.transcript_download, name='transcript_download'),
]

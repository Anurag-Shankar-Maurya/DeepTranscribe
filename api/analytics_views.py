"""
Enhanced views for user analytics and detailed transcript data
"""

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Count, Sum, Avg, Q
from django.db.models.functions import TruncDate, TruncMonth
from core.models import Transcript, TranscriptSegment, TranscriptEmbedding
from api.models import ChatMessage
from datetime import datetime, timedelta
from collections import defaultdict
import json


@login_required
def user_dashboard(request):
    """Enhanced user dashboard with comprehensive analytics"""
    user = request.user
    
    # Get transcript statistics
    total_transcripts = Transcript.objects.filter(user=user).count()
    total_segments = TranscriptSegment.objects.filter(transcript__user=user).count()
    total_duration = Transcript.objects.filter(user=user).aggregate(
        total=Sum('duration')
    )['total'] or 0
    
    # Get recent transcripts
    recent_transcripts = Transcript.objects.filter(user=user).order_by('-created_at')[:5]
    
    # Get transcripts by date (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    transcripts_by_date = Transcript.objects.filter(
        user=user,
        created_at__gte=thirty_days_ago
    ).annotate(
        date=TruncDate('created_at')
    ).values('date').annotate(
        count=Count('id')
    ).order_by('date')
    
    # Get speaker statistics
    speaker_stats = TranscriptSegment.objects.filter(
        transcript__user=user,
        speaker__isnull=False
    ).values('speaker').annotate(
        segment_count=Count('id'),
        total_time=Sum('end_time') - Sum('start_time')
    ).order_by('-segment_count')[:10]
    
    # Get chat message statistics
    total_chat_messages = ChatMessage.objects.filter(user=user).count()
    user_messages = ChatMessage.objects.filter(user=user, role='user').count()
    assistant_messages = ChatMessage.objects.filter(user=user, role='assistant').count()
    
    context = {
        'total_transcripts': total_transcripts,
        'total_segments': total_segments,
        'total_duration_minutes': int(total_duration / 60),
        'total_duration_hours': round(total_duration / 3600, 1),
        'recent_transcripts': recent_transcripts,
        'transcripts_by_date': list(transcripts_by_date),
        'speaker_stats': list(speaker_stats),
        'total_chat_messages': total_chat_messages,
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
    }
    
    return render(request, 'api/user_dashboard.html', context)


@login_required
def transcript_detail_enhanced(request, pk):
    """Enhanced transcript detail with minute-by-minute breakdown"""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    segments = transcript.segments.all().order_by('start_time')
    
    # Calculate minute-by-minute statistics
    minute_data = defaultdict(lambda: {
        'segments': [],
        'speakers': set(),
        'total_words': 0,
        'duration': 0
    })
    
    for segment in segments:
        minute = int(segment.start_time // 60)
        minute_data[minute]['segments'].append({
            'text': segment.text,
            'speaker': segment.speaker,
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'confidence': segment.confidence
        })
        minute_data[minute]['speakers'].add(segment.speaker if segment.speaker else 'Unknown')
        minute_data[minute]['total_words'] += len(segment.text.split())
        minute_data[minute]['duration'] += (segment.end_time - segment.start_time)
    
    # Convert sets to lists for JSON serialization
    for minute, data in minute_data.items():
        data['speakers'] = list(data['speakers'])
    
    # Get speaker distribution
    speaker_distribution = segments.values('speaker').annotate(
        count=Count('id'),
        total_duration=Sum('end_time') - Sum('start_time')
    ).order_by('-count')
    
    # Calculate average confidence
    avg_confidence = segments.aggregate(Avg('confidence'))['confidence__avg'] or 0
    
    # Get embeddings count
    embeddings_count = TranscriptEmbedding.objects.filter(transcript=transcript).count()
    
    context = {
        'transcript': transcript,
        'segments': segments,
        'minute_data': dict(minute_data),
        'speaker_distribution': list(speaker_distribution),
        'avg_confidence': round(avg_confidence, 2),
        'embeddings_count': embeddings_count,
        'total_minutes': int(transcript.duration / 60),
        'total_words': sum(len(s.text.split()) for s in segments),
    }
    
    return render(request, 'api/transcript_detail_enhanced.html', context)


@login_required
def user_transcripts_api(request):
    """API endpoint for user transcripts with filters"""
    user = request.user
    
    # Get query parameters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    search_query = request.GET.get('search', '')
    
    # Build query
    transcripts = Transcript.objects.filter(user=user)
    
    if start_date:
        transcripts = transcripts.filter(created_at__gte=start_date)
    
    if end_date:
        transcripts = transcripts.filter(created_at__lte=end_date)
    
    if search_query:
        transcripts = transcripts.filter(
            Q(title__icontains=search_query) | 
            Q(segments__text__icontains=search_query)
        ).distinct()
    
    # Prepare response data
    data = []
    for transcript in transcripts.order_by('-created_at'):
        data.append({
            'id': transcript.id,
            'title': transcript.title,
            'created_at': transcript.created_at.isoformat(),
            'updated_at': transcript.updated_at.isoformat(),
            'duration': transcript.duration,
            'duration_formatted': f"{transcript.duration // 60}m {transcript.duration % 60}s",
            'is_complete': transcript.is_complete,
            'segment_count': transcript.segments.count(),
        })
    
    return JsonResponse({
        'transcripts': data,
        'count': len(data)
    })


@login_required
def transcript_analytics_api(request, pk):
    """API endpoint for detailed transcript analytics"""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    segments = transcript.segments.all()
    
    # Speaker timeline
    speaker_timeline = []
    for segment in segments.order_by('start_time'):
        speaker_timeline.append({
            'speaker': segment.speaker if segment.speaker else 0,
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'duration': segment.end_time - segment.start_time,
            'text_preview': segment.text[:50] + '...' if len(segment.text) > 50 else segment.text
        })
    
    # Word frequency
    all_text = ' '.join([s.text for s in segments])
    words = all_text.lower().split()
    word_freq = {}
    for word in words:
        # Clean word
        word = word.strip('.,!?;:"\'')
        if len(word) > 3 and word not in ['that', 'this', 'with', 'from', 'have', 'been', 'were', 'said']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Speaker statistics
    speaker_stats = {}
    for segment in segments:
        speaker_id = segment.speaker if segment.speaker else 'Unknown'
        if speaker_id not in speaker_stats:
            speaker_stats[speaker_id] = {
                'segment_count': 0,
                'total_duration': 0,
                'word_count': 0,
                'avg_confidence': []
            }
        speaker_stats[speaker_id]['segment_count'] += 1
        speaker_stats[speaker_id]['total_duration'] += (segment.end_time - segment.start_time)
        speaker_stats[speaker_id]['word_count'] += len(segment.text.split())
        speaker_stats[speaker_id]['avg_confidence'].append(segment.confidence)
    
    # Calculate average confidence per speaker
    for speaker_id, stats in speaker_stats.items():
        if stats['avg_confidence']:
            stats['avg_confidence'] = sum(stats['avg_confidence']) / len(stats['avg_confidence'])
        else:
            stats['avg_confidence'] = 0
    
    return JsonResponse({
        'transcript': {
            'id': transcript.id,
            'title': transcript.title,
            'duration': transcript.duration,
            'created_at': transcript.created_at.isoformat()
        },
        'speaker_timeline': speaker_timeline,
        'top_words': top_words,
        'speaker_stats': speaker_stats,
        'total_segments': segments.count(),
        'total_words': len(words)
    })

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from .models import Transcript, TranscriptSegment, TranscriptSettings
import json
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Path to a Unicode font like NotoSans (adjust as necessary)
FONT_PATH = os.path.join('fonts', 'NotoSans-Regular.ttf')

def index(request):
    """Render the index page."""
    return render(request, 'core/index.html')


@login_required
def transcribe(request):
    """Render the transcription page."""
    return render(request, 'core/transcribe.html')


@login_required
def transcript_list(request):
    """Render the list of user transcripts."""
    transcripts = Transcript.objects.filter(user=request.user)
    return render(request, 'core/transcript_list.html', {'transcripts': transcripts})


@login_required
def transcript_detail(request, pk):
    """Render the transcript detail page."""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    segments = transcript.segments.all()
    return render(request, 'core/transcript_detail.html', {
        'transcript': transcript,
        'segments': segments
    })


@login_required
def transcript_edit(request, pk):
    """Render the transcript edit page."""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    
    if request.method == 'POST':
        # Process form data
        title = request.POST.get('title', '')
        if title:
            transcript.title = title
            transcript.save()
            return redirect('core:transcript_detail', pk=transcript.pk)
    
    return render(request, 'core/transcript_edit.html', {'transcript': transcript})


@login_required
def transcript_delete(request, pk):
    """Delete a transcript."""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    
    if request.method == 'POST':
        transcript.delete()
        return redirect('core:transcript_list')
    
    return render(request, 'core/transcript_delete.html', {'transcript': transcript})


@login_required
def transcript_download(request, pk, format):
    """Download the transcript in the requested format (pdf, txt, json)."""
    transcript = get_object_or_404(Transcript, pk=pk, user=request.user)
    segments = transcript.segments.all()

    if format == 'json':
        # Serialize transcript and segments to JSON
        transcript_data = {
            'id': transcript.id,
            'title': transcript.title,
            'created_at': transcript.created_at.isoformat(),
            'duration': transcript.duration,
            'is_complete': transcript.is_complete,
            'settings': {
                'model': transcript.settings.model,
                'language': transcript.settings.language,
                'diarize': transcript.settings.diarize,
                'punctuate': transcript.settings.punctuate,
                'numerals': transcript.settings.numerals,
                'smart_format': transcript.settings.smart_format,
            },
            'segments': [
                {
                    'speaker': segment.speaker,
                    'text': segment.text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'confidence': segment.confidence,
                }
                for segment in segments
            ]
        }
        response = HttpResponse(json.dumps(transcript_data, indent=2), content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="{transcript.title}.json"'
        return response

    elif format == 'txt':
        # Create plain text representation
        lines = [f"Transcript: {transcript.title}\n"]
        for segment in segments:
            speaker = f"Speaker {segment.speaker}" if segment.speaker is not None else "Unknown Speaker"
            lines.append(f"{speaker} [{segment.start_time:.2f}s - {segment.end_time:.2f}s]:\n{segment.text}\n")
        text_content = "\n".join(lines)
        response = HttpResponse(text_content, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="{transcript.title}.txt"'
        return response

    elif format == 'pdf':
        # Generate PDF using ReportLab
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - 50

        # Register a Unicode font (NotoSans)
        pdfmetrics.registerFont(TTFont('NotoSans', FONT_PATH))

        p.setFont("NotoSans", 16)
        p.drawString(50, y, f"Transcript: {transcript.title}")
        y -= 30
        p.setFont("NotoSans", 12)

        for segment in segments:
            speaker = f"Speaker {segment.speaker}" if segment.speaker is not None else "Unknown Speaker"
            text = f"{speaker} [{segment.start_time:.2f}s - {segment.end_time:.2f}s]: {segment.text}"

            # Split long lines
            lines = []
            while len(text) > 90:
                split_index = text.rfind(' ', 0, 90)
                if split_index == -1:
                    split_index = 90
                lines.append(text[:split_index])
                text = text[split_index + 1:]
            lines.append(text)

            for line in lines:
                if y < 50:
                    p.showPage()
                    y = height - 50
                    p.setFont("NotoSans", 12)
                p.drawString(50, y, line)
                y -= 15
            y -= 10

        p.save()
        pdf = buffer.getvalue()
        buffer.close()
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{transcript.title}.pdf"'
        return response

    else:
        return HttpResponseBadRequest("Invalid format requested.")

from django import forms
from django.forms import inlineformset_factory
from .models import Transcript, TranscriptSegment, TranscriptSettings

class TranscriptForm(forms.ModelForm):
    class Meta:
        model = Transcript
        fields = ['title'] # Only allow editing the title for now

class TranscriptSettingsForm(forms.ModelForm):
    class Meta:
        model = TranscriptSettings
        fields = ['model', 'language', 'diarize', 'punctuate', 'numerals', 'smart_format']

# Define a form for individual transcript segments
class TranscriptSegmentForm(forms.ModelForm):
    class Meta:
        model = TranscriptSegment
        fields = ['text', 'speaker', 'start_time', 'end_time', 'confidence']

# Create a formset for TranscriptSegmentForm
# This allows managing multiple TranscriptSegment objects
TranscriptSegmentFormSet = inlineformset_factory(
    Transcript,
    TranscriptSegment,
    form=TranscriptSegmentForm,
    extra=0,  # Do not show extra empty forms by default, only existing ones
    can_delete=True # Allow deleting segments
)

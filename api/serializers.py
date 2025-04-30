from rest_framework import serializers
from core.models import Transcript, TranscriptSegment, TranscriptSettings


class TranscriptSettingsSerializer(serializers.ModelSerializer):
    """Serializer for transcript settings."""
    
    class Meta:
        model = TranscriptSettings
        fields = ['model', 'language', 'diarize', 'punctuate', 'numerals', 'smart_format']


class TranscriptSegmentSerializer(serializers.ModelSerializer):
    """Serializer for transcript segments."""
    
    class Meta:
        model = TranscriptSegment
        fields = ['id', 'transcript', 'text', 'speaker', 'start_time', 'end_time', 'confidence', 'created_at']
        read_only_fields = ['created_at']


class TranscriptSerializer(serializers.ModelSerializer):
    """Serializer for transcripts."""
    settings = TranscriptSettingsSerializer(required=False)
    segments = serializers.SerializerMethodField()
    
    class Meta:
        model = Transcript
        fields = ['id', 'title', 'created_at', 'updated_at', 'user', 'is_complete', 'duration', 'settings', 'segments']
        read_only_fields = ['created_at', 'updated_at', 'user']
    
    def get_segments(self, obj):
        """Get the most recent segments (limit to 10)."""
        segments = obj.segments.all().order_by('start_time')[:10]
        return TranscriptSegmentSerializer(segments, many=True).data
    
    def create(self, validated_data):
        """Create a transcript with settings."""
        settings_data = validated_data.pop('settings', None)
        transcript = Transcript.objects.create(**validated_data)
        
        if settings_data:
            TranscriptSettings.objects.create(transcript=transcript, **settings_data)
        else:
            TranscriptSettings.objects.create(transcript=transcript)
        
        return transcript
    
    def update(self, instance, validated_data):
        """Update a transcript with settings."""
        settings_data = validated_data.pop('settings', None)
        
        # Update transcript fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        # Update settings if provided
        if settings_data and hasattr(instance, 'settings'):
            settings = instance.settings
            for attr, value in settings_data.items():
                setattr(settings, attr, value)
            settings.save()
        
        return instance
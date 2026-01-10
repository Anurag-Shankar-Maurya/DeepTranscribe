from django.contrib import admin
from .models import Transcript, TranscriptSegment, TranscriptSettings, TranscriptEmbedding


class TranscriptSegmentInline(admin.TabularInline):
    model = TranscriptSegment
    extra = 0


class TranscriptSettingsInline(admin.StackedInline):
    model = TranscriptSettings
    can_delete = False


@admin.register(Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at', 'is_complete', 'duration')
    list_filter = ('is_complete', 'created_at')
    search_fields = ('title', 'user__username')
    inlines = [TranscriptSettingsInline, TranscriptSegmentInline]


@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = ('transcript', 'speaker', 'text', 'start_time', 'end_time')
    list_filter = ('speaker', 'transcript')
    search_fields = ('text', 'transcript__title')


@admin.register(TranscriptSettings)
class TranscriptSettingsAdmin(admin.ModelAdmin):
    list_display = ('transcript', 'model', 'language', 'diarize')
    list_filter = ('model', 'language', 'diarize')
    search_fields = ('transcript__title',)


@admin.register(TranscriptEmbedding)
class TranscriptEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('transcript', 'segment', 'created_at')
    list_filter = ('created_at', 'transcript')
    search_fields = ('transcript__title', 'segment__text')
    readonly_fields = ('created_at', 'embedding')

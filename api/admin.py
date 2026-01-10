from django.contrib import admin
from .models import (
    ChatSession, ChatMessage, ChatMessageEmbedding,
    Entity, KnowledgeTriple, HierarchicalSummary
)


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('timestamp',)


class ChatMessageEmbeddingInline(admin.StackedInline):
    model = ChatMessageEmbedding
    can_delete = False
    readonly_fields = ('created_at',)


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'user', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('session_id', 'user__username', 'summary')
    readonly_fields = ('created_at', 'updated_at')
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'session', 'role', 'content_short', 'timestamp')
    list_filter = ('role', 'timestamp', 'session')
    search_fields = ('content', 'user__username', 'session__session_id')
    readonly_fields = ('timestamp',)
    inlines = [ChatMessageEmbeddingInline]

    def content_short(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_short.short_description = 'Content'


@admin.register(ChatMessageEmbedding)
class ChatMessageEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('chat_message', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('chat_message__content', 'chat_message__user__username')
    readonly_fields = ('created_at',)


class KnowledgeTripleInline(admin.TabularInline):
    model = KnowledgeTriple
    extra = 0
    fk_name = 'subject'


@admin.register(Entity)
class EntityAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'entity_type', 'last_mentioned')
    list_filter = ('entity_type', 'last_mentioned')
    search_fields = ('name', 'description', 'user__username')
    readonly_fields = ('last_mentioned',)
    inlines = [KnowledgeTripleInline]


@admin.register(KnowledgeTriple)
class KnowledgeTripleAdmin(admin.ModelAdmin):
    list_display = ('subject', 'predicate', 'object_entity', 'user', 'weight')
    list_filter = ('predicate', 'weight', 'subject__entity_type', 'object_entity__entity_type')
    search_fields = ('subject__name', 'predicate', 'object_entity__name', 'user__username')
    readonly_fields = ('source_message',)


@admin.register(HierarchicalSummary)
class HierarchicalSummaryAdmin(admin.ModelAdmin):
    list_display = ('user', 'level', 'start_date', 'end_date', 'content_short')
    list_filter = ('level', 'start_date', 'end_date')
    search_fields = ('content', 'user__username')
    readonly_fields = ('embedding',)

    def content_short(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_short.short_description = 'Content'

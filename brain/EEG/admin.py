from django.contrib import admin
from .models import EEGSession, EmotionPrediction, JournalEntry, ChatMessage

@admin.register(EEGSession)
class EEGSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status', 'created_at']
    list_filter  = ['status']

@admin.register(EmotionPrediction)
class EmotionPredictionAdmin(admin.ModelAdmin):
    list_display = ['emotion', 'valence', 'arousal', 'dominance', 'distress_flag', 'created_at']
    list_filter  = ['valence', 'arousal', 'distress_flag']

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ['user', 'emotion', 'valence', 'distress', 'created_at']

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'role', 'created_at']

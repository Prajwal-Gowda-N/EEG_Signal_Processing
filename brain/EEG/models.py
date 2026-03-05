from django.db import models
from django.contrib.auth.models import User
import uuid


class EEGSession(models.Model):
    """One complete EEG analysis session."""

    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('complete',   'Complete'),
        ('failed',     'Failed'),
    ]

    id            = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Uploaded file
    eeg_file      = models.FileField(upload_to='eeg_uploads/', null=True, blank=True)
    file_name     = models.CharField(max_length=255, blank=True)

    # Session metadata
    notes         = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Session {str(self.id)[:8]} — {self.user.username} — {self.status}"


class EmotionPrediction(models.Model):
    """Emotion prediction result for one session."""

    VALENCE_CHOICES   = [('High', 'High'), ('Low', 'Low'), ('Unknown', 'Unknown')]
    AROUSAL_CHOICES   = [('High', 'High'), ('Low', 'Low'), ('Unknown', 'Unknown')]
    DOMINANCE_CHOICES = [('High', 'High'), ('Low', 'Low'), ('Unknown', 'Unknown')]

    session         = models.OneToOneField(EEGSession, on_delete=models.CASCADE,
                                           related_name='prediction')
    created_at      = models.DateTimeField(auto_now_add=True)

    # Raw prediction
    emotion         = models.CharField(max_length=100, default='Unknown')
    summary         = models.CharField(max_length=255, blank=True)
    valence         = models.CharField(max_length=10, choices=VALENCE_CHOICES,   default='Unknown')
    arousal         = models.CharField(max_length=10, choices=AROUSAL_CHOICES,   default='Unknown')
    dominance       = models.CharField(max_length=10, choices=DOMINANCE_CHOICES, default='Unknown')

    # Confidence scores
    valence_conf    = models.FloatField(default=0.0)
    arousal_conf    = models.FloatField(default=0.0)
    dominance_conf  = models.FloatField(default=0.0)

    # Agent output
    explanation     = models.TextField(blank=True)
    action_steps    = models.JSONField(default=list)
    meditation      = models.TextField(blank=True)
    referral        = models.TextField(blank=True)
    distress_flag   = models.BooleanField(default=False)
    sources         = models.JSONField(default=list)
    rag_quality     = models.IntegerField(default=0)

    # Suggested follow-up questions from agent
    followup_questions = models.JSONField(default=list)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.emotion} ({self.valence}V/{self.arousal}A) — {self.session}"

    @property
    def emotion_emoji(self):
        emoji_map = {
            'Relaxed / Content'  : '😌',
            'Happy / Excited'    : '😄',
            'Calm / Serene'      : '🧘',
            'Pleased / Joyful'   : '😊',
            'Anxious / Nervous'  : '😰',
            'Stressed / Tense'   : '😤',
            'Sad / Depressed'    : '😢',
            'Bored / Fatigued'   : '😴',
        }
        return emoji_map.get(self.emotion, '🧠')

    @property
    def valence_color(self):
        return '#4ade80' if self.valence == 'High' else '#f87171'

    @property
    def arousal_color(self):
        return '#60a5fa' if self.arousal == 'High' else '#a78bfa'


class JournalEntry(models.Model):
    """One entry in the emotion journal — logged per session."""

    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='journal')
    session     = models.OneToOneField(EEGSession, on_delete=models.CASCADE,
                                       related_name='journal_entry', null=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    emotion     = models.CharField(max_length=100)
    valence     = models.CharField(max_length=10)
    arousal     = models.CharField(max_length=10)
    dominance   = models.CharField(max_length=10)
    distress    = models.BooleanField(default=False)

    # Short personal note the user can add
    user_note   = models.TextField(blank=True)

    # Agent summary for this entry
    agent_summary = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} — {self.emotion} — {self.created_at.date()}"


class ChatMessage(models.Model):
    """One message in a follow-up chat session."""

    ROLE_CHOICES = [('user', 'User'), ('agent', 'Agent')]

    session    = models.ForeignKey(EEGSession, on_delete=models.CASCADE,
                                   related_name='chat_messages')
    created_at = models.DateTimeField(auto_now_add=True)
    role       = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content    = models.TextField()

    # Agent metadata (only for agent messages)
    sources    = models.JSONField(default=list)
    rag_quality= models.IntegerField(default=0)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"

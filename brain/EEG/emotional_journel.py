"""
Emotion Journal
================
Persists emotion predictions + explanations across sessions.
Used by the agent for mood trend tracking and distress detection.

Storage: JSON file  (simple, no database needed for now)
         → Django will later use this as a backend service

Features:
  - Log every prediction + explanation
  - Trend analysis (last N sessions)
  - Distress detection (persistent low valence + high arousal)
  - Mood summary for agent context
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from collections import Counter

JOURNAL_DIR  = "rag/journal"
JOURNAL_FILE = os.path.join(JOURNAL_DIR, "emotion_journal.json")
os.makedirs(JOURNAL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DISTRESS RULES
# ─────────────────────────────────────────────

# If these combinations appear >= DISTRESS_THRESHOLD times
# in the last DISTRESS_WINDOW sessions → flag for referral
DISTRESS_WINDOW    = 5    # look at last 5 sessions
DISTRESS_THRESHOLD = 3    # 3 out of 5 sessions in distress = flag

DISTRESS_PATTERNS = [
    # (valence, arousal, dominance) — all Low/High combinations indicating distress
    ('Low', 'High', 'Low'),   # Anxious / Nervous
    ('Low', 'Low',  'Low'),   # Bored / Fatigued / Depressed
    ('Low', 'High', 'High'),  # Stressed / Tense
]

# Positive patterns  (used for mood trend)
POSITIVE_PATTERNS = [
    ('High', 'High', 'High'),  # Relaxed / Content
    ('High', 'Low',  'High'),  # Happy / Excited
    ('High', 'High', 'Low'),   # Pleased
]


# ─────────────────────────────────────────────
# JOURNAL CLASS
# ─────────────────────────────────────────────

class EmotionJournal:
    """
    Loads and saves emotion session records.
    Each record = one EEG analysis session.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id      = user_id
        self.journal_path = os.path.join(
            JOURNAL_DIR, f"journal_{user_id}.json")
        self.records      = self._load()

    def _load(self) -> list:
        if os.path.exists(self.journal_path):
            with open(self.journal_path) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.journal_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    # ─────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────

    def log(self, prediction: dict, explanation: str,
            session_id: str = None) -> dict:
        """
        Log one emotion session to the journal.

        Parameters
        ──────────
        prediction  : dict from emotion_inference.predict()
        explanation : agent-generated explanation string
        session_id  : optional session UUID

        Returns
        ───────
        The record that was saved.
        """
        record = {
            'id'          : session_id or str(uuid.uuid4()),
            'user_id'     : self.user_id,
            'timestamp'   : datetime.now().isoformat(),
            'date'        : datetime.now().strftime('%Y-%m-%d'),
            'time'        : datetime.now().strftime('%H:%M'),

            # Prediction
            'emotion'     : prediction.get('emotion',   'Unknown'),
            'summary'     : prediction.get('summary',   ''),
            'valence'     : prediction.get('valence',   {}).get('label', 'Unknown'),
            'arousal'     : prediction.get('arousal',   {}).get('label', 'Unknown'),
            'dominance'   : prediction.get('dominance', {}).get('label', 'Unknown'),
            'valence_conf': prediction.get('valence',   {}).get('confidence', 0.0),
            'arousal_conf': prediction.get('arousal',   {}).get('confidence', 0.0),
            'dom_conf'    : prediction.get('dominance', {}).get('confidence', 0.0),

            # Agent output
            'explanation' : explanation[:500],   # truncated for storage
            'distress'    : self._is_distress(prediction),
        }

        self.records.append(record)
        self._save()
        return record

    # ─────────────────────────────────────────
    # DISTRESS DETECTION
    # ─────────────────────────────────────────

    def _is_distress(self, prediction: dict) -> bool:
        """Check if a single prediction matches a distress pattern."""
        v = prediction.get('valence',   {}).get('label', '')
        a = prediction.get('arousal',   {}).get('label', '')
        d = prediction.get('dominance', {}).get('label', '')
        return (v, a, d) in DISTRESS_PATTERNS

    def check_distress_trend(self) -> dict:
        """
        Analyse last DISTRESS_WINDOW sessions for persistent distress.

        Returns
        ───────
        {
          'flagged': bool,
          'distress_count': int,
          'window': int,
          'message': str,
          'suggest_referral': bool,
        }
        """
        recent = self.records[-DISTRESS_WINDOW:]
        if not recent:
            return {'flagged': False, 'distress_count': 0,
                    'window': 0, 'message': '', 'suggest_referral': False}

        distress_count = sum(1 for r in recent if r.get('distress', False))
        flagged        = distress_count >= DISTRESS_THRESHOLD

        message = ""
        if flagged:
            message = (
                f"You've shown signs of emotional distress in "
                f"{distress_count} of your last {len(recent)} sessions. "
                f"It may be helpful to speak with a mental health professional."
            )

        return {
            'flagged'         : flagged,
            'distress_count'  : distress_count,
            'window'          : len(recent),
            'message'         : message,
            'suggest_referral': flagged,
        }

    # ─────────────────────────────────────────
    # MOOD TREND
    # ─────────────────────────────────────────

    def get_mood_trend(self, last_n: int = 10) -> dict:
        """
        Analyse mood trend across last N sessions.

        Returns
        ───────
        {
          'total_sessions': int,
          'positive_pct'  : float,
          'distress_pct'  : float,
          'most_common'   : str,
          'trend'         : 'improving' | 'declining' | 'stable',
          'summary'       : str,
          'by_emotion'    : {emotion: count},
        }
        """
        recent = self.records[-last_n:]
        if not recent:
            return {'total_sessions': 0, 'summary': 'No sessions recorded yet.'}

        emotions = [r['emotion'] for r in recent]
        by_emotion = dict(Counter(emotions).most_common())
        most_common = emotions[0] if emotions else 'Unknown'

        positive_count = sum(
            1 for r in recent
            if (r['valence'], r['arousal'], r['dominance']) in POSITIVE_PATTERNS)
        distress_count = sum(1 for r in recent if r.get('distress', False))

        positive_pct = round(positive_count / len(recent) * 100, 1)
        distress_pct = round(distress_count / len(recent) * 100, 1)

        # Trend: compare first half vs second half
        mid     = len(recent) // 2
        first_h = sum(1 for r in recent[:mid]  if r.get('distress', False))
        second_h= sum(1 for r in recent[mid:]  if r.get('distress', False))

        if   second_h < first_h: trend = 'improving'
        elif second_h > first_h: trend = 'declining'
        else:                     trend = 'stable'

        # Build summary string for agent context
        summary = (
            f"Over your last {len(recent)} sessions: "
            f"{positive_pct}% positive, {distress_pct}% distress signals. "
            f"Most common emotion: {most_common}. "
            f"Mood trend: {trend}."
        )

        return {
            'total_sessions': len(recent),
            'positive_pct'  : positive_pct,
            'distress_pct'  : distress_pct,
            'most_common'   : most_common,
            'trend'         : trend,
            'trend_arrow'   : {'improving': '↑', 'declining': '↓', 'stable': '→'}[trend],
            'summary'       : summary,
            'by_emotion'    : by_emotion,
        }

    # ─────────────────────────────────────────
    # CONVERSATION SUMMARY
    # ─────────────────────────────────────────

    def summarize_for_agent(self) -> str:
        """
        Return a short string the agent can inject as context
        to be aware of the user's emotional history.
        """
        if len(self.records) < 2:
            return "This is one of the user's first sessions."

        trend    = self.get_mood_trend(last_n=10)
        distress = self.check_distress_trend()

        lines = [
            f"User has had {len(self.records)} total sessions.",
            trend['summary'],
        ]
        if distress['flagged']:
            lines.append(
                f"⚠️ Distress alert: {distress['distress_count']} distress "
                f"signals in last {distress['window']} sessions.")

        return " ".join(lines)

    # ─────────────────────────────────────────
    # JOURNAL EXPORT  (for Django view)
    # ─────────────────────────────────────────

    def get_recent(self, n: int = 20) -> list:
        """Return last N records for display in journal page."""
        return list(reversed(self.records[-n:]))

    def get_stats(self) -> dict:
        """Full statistics for dashboard display."""
        if not self.records:
            return {'total': 0}

        trend    = self.get_mood_trend()
        distress = self.check_distress_trend()

        return {
            'total'          : len(self.records),
            'first_session'  : self.records[0]['date'] if self.records else None,
            'last_session'   : self.records[-1]['date'] if self.records else None,
            'trend'          : trend,
            'distress_alert' : distress,
        }
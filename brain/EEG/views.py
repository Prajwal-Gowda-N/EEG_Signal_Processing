import os
import json
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import TruncDate

from .models import EEGSession, EmotionPrediction, JournalEntry, ChatMessage
from .forms import EEGUploadForm, RegisterForm, LoginForm, JournalNoteForm, ChatForm



# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def load_emotion_predictor():
    """Lazy-load EmotionPredictor (heavy model, load once)."""
    import sys
    sys.path.insert(0, str(settings.BASE_DIR))
    from .emotion_inference import EmotionPredictor
    return EmotionPredictor(model_dir=settings.EMOTION_MODEL_DIR)


def load_agent():
    """Lazy-load EmotionAgent (heavy, load once)."""
    import sys
    sys.path.insert(0, str(settings.BASE_DIR))
    os.environ.setdefault('GOOGLE_API_KEY', settings.GOOGLE_API_KEY)
    os.environ.setdefault('LLM_BACKEND',    settings.LLM_BACKEND)
    from .rag_agents import EmotionAgent
    return EmotionAgent()


_predictor = None
_agent     = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_emotion_predictor()
    return _predictor


def get_agent():
    global _agent
    if _agent is None:
        _agent = load_agent()
    return _agent


def parse_eeg_file(file_obj, file_name: str) -> np.ndarray:
    """
    Parse uploaded EEG file → numpy array (samples, 14).
    Supports .csv, .txt, .npy
    """
    ext = os.path.splitext(file_name)[1].lower()

    if ext == '.npy':
        data = np.load(file_obj)
    elif ext in ('.csv', '.txt'):
        import io
        content = file_obj.read().decode('utf-8')
        data    = np.loadtxt(io.StringIO(content), delimiter=',')
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Ensure 14 channels
    if data.shape[1] > 14:
        data = data[:, :14]
    elif data.shape[1] < 14:
        pad  = np.zeros((data.shape[0], 14 - data.shape[1]), dtype=np.float32)
        data = np.hstack([data, pad])

    return data.astype(np.float32)



def get_dashboard_stats(user=None):
    if user and user.is_authenticated:
        sessions = EEGSession.objects.filter(user=user, status='complete')
        journal  = JournalEntry.objects.filter(user=user)
    else:
        sessions = EEGSession.objects.filter(status='complete')
        journal  = JournalEntry.objects.all()

    total_sessions = sessions.count()
    distress_count = journal.filter(distress=True).count()
    positive_count = journal.filter(valence='High').count()

    # Debug prints — remove later
    print(f"[DEBUG STATS] User: {user}, Total sessions: {total_sessions}")
    print(f"[DEBUG STATS] Distress: {distress_count}, Positive valence: {positive_count}")
    print(f"[DEBUG STATS] Journal entries for user: {journal.count()}")

    emotions = list(
        EmotionPrediction.objects
        .filter(session__status='complete')
        .values('emotion')
        .annotate(count=Count('emotion'))
        .order_by('-count')
    )

    recent_preds = EmotionPrediction.objects.filter(session__status='complete').order_by('-created_at')[:10]
    trend_data = [
        {
            'date': p.created_at.strftime('%b %d'),
            'valence': 1 if p.valence == 'High' else 0,
            'arousal': 1 if p.arousal == 'High' else 0,
            'emotion': p.emotion,
        }
        for p in reversed(list(recent_preds))
    ]

    return {
        'total_sessions': total_sessions,
        'distress_count': distress_count,
        'positive_count': positive_count,
        'emotions': emotions,
        'trend_data': json.dumps(trend_data),
    }

# ─────────────────────────────────────────────
# AUTH VIEWS (optional - kept for demo purposes)
# ─────────────────────────────────────────────


def landing(request):
    return render(request, 'eeg_app/landing.html')


def register_view(request):
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, f'Welcome, {user.username}! Your account is ready.')
        return redirect('dashboard')
    return render(request, 'eeg_app/auth.html', {'form': form, 'mode': 'register'})


def login_view(request):
    form = LoginForm(request, request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect(request.GET.get('next', 'dashboard'))
    return render(request, 'eeg_app/auth.html', {'form': form, 'mode': 'login'})


def logout_view(request):
    logout(request)
    return redirect('landing')



# ─────────────────────────────────────────────
# MAIN PAGES (NO AUTH REQUIRED)
# ─────────────────────────────────────────────


def dashboard(request):
    user = request.user if request.user.is_authenticated else None
    stats    = get_dashboard_stats(user)
    sessions = EEGSession.objects.filter(user=user)[:8] if user else EEGSession.objects.filter(status='complete')[:8]
    latest   = (EmotionPrediction.objects
                .filter(session__status='complete')
                .order_by('-created_at').first())
    ctx = {**stats, 'sessions': sessions, 'latest': latest}
    return render(request, 'eeg_app/dashboard.html', ctx)


def upload(request):
    form = EEGUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        f    = request.FILES['eeg_file']
        note = form.cleaned_data.get('notes', '')

        # Create session record (use anonymous user or create temp user)
        user = request.user if request.user.is_authenticated else None
        session = EEGSession.objects.create(
            user      = user,
            eeg_file  = f,
            file_name = f.name,
            notes     = note,
            status    = 'processing',
        )

        try:
            # Parse EEG
            f.seek(0)
            eeg = parse_eeg_file(f, f.name)

            # Predict emotion
            predictor  = get_predictor()
            prediction = predictor.predict(eeg, preprocess=True)

            # Run agent
            agent  = get_agent()
            result = agent.explain(prediction, user_id=str(request.user.id) if user else 'anonymous',
                                   session_id=str(session.id))

            # Save prediction to DB
            pred = EmotionPrediction.objects.create(
                session            = session,
                emotion            = prediction.get('emotion', 'Unknown'),
                summary            = prediction.get('summary', ''),
                valence            = prediction['valence']['label'],
                arousal            = prediction['arousal']['label'],
                dominance          = prediction['dominance']['label'],
                valence_conf       = prediction['valence']['confidence'],
                arousal_conf       = prediction['arousal']['confidence'],
                dominance_conf     = prediction['dominance']['confidence'],
                explanation        = result.get('explanation', ''),
                action_steps       = result.get('action_steps', []),
                meditation         = result.get('meditation', ''),
                referral           = result.get('referral', ''),
                distress_flag      = result.get('distress_flag', False),
                sources            = result.get('sources', []),
                rag_quality        = result.get('rag_quality_score', 0),
                followup_questions = result.get('followup_questions', []),
            )

            # Save journal entry (only if user exists)
            if user:
                journal_entry = JournalEntry.objects.create(
                        user          = user,
                        session       = session,
                        emotion       = pred.emotion,
                        valence       = pred.valence,
                        arousal       = pred.arousal,
                        dominance     = pred.dominance,
                        distress      = pred.distress_flag,
                        agent_summary = result.get('explanation', '')[:300],
                )
                print(f"[JOURNAL DEBUG] Created entry for {user.username}:")
                print(f"  - Valence: {pred.valence}")
                print(f"  - Distress flag: {pred.distress_flag}")
                print(f"  - Emotion: {pred.emotion}")

            # Save first agent message
            ChatMessage.objects.create(
                session     = session,
                role        = 'agent',
                content     = result.get('explanation', ''),
                sources     = result.get('sources', []),
                rag_quality = result.get('rag_quality_score', 0),
            )

            session.status = 'complete'
            session.save()

            messages.success(request, 'EEG analysed successfully!')
            return redirect('results', session_id=str(session.id))

        except Exception as e:
            session.status = 'failed'
            session.save()
            messages.error(request, f'Analysis failed: {str(e)}')

    return render(request, 'eeg_app/upload.html', {'form': form})


def results(request, session_id):
    session    = get_object_or_404(EEGSession, id=session_id)
    prediction = get_object_or_404(EmotionPrediction, session=session)
    chat_form  = ChatForm()
    messages_qs = ChatMessage.objects.filter(session=session)

    ctx = {
        'session'   : session,
        'prediction': prediction,
        'chat_form' : chat_form,
        'messages'  : messages_qs,
    }
    return render(request, 'eeg_app/results.html', ctx)


def journal(request):
    user = request.user if request.user.is_authenticated else None
    entries = JournalEntry.objects.filter(user=user) if user else JournalEntry.objects.all()

    total = entries.count()
    distress = entries.filter(distress=True).count()
    positive = entries.filter(valence='High').count()

    # Emotion count (Top 6)
    emotion_counts = list(
        entries.values('emotion')
        .annotate(count=Count('emotion'))
        .order_by('-count')[:6]
    )

    # Last 30 days trend (safe & modern way)
    since = timezone.now() - timedelta(days=30)

    daily_data = list(
        entries
        .filter(created_at__gte=since)
        .annotate(day=TruncDate('created_at'))
        .values('day', 'valence')
        .order_by('day')
    )

    context = {
        'entries': entries.order_by('-created_at')[:30],
        'total': total,
        'distress_count': distress,
        'positive_count': positive,
        'emotion_counts': emotion_counts,   # safe list
        'daily_data': daily_data,           # safe list
    }

    return render(request, 'eeg_app/journal.html', context)

def chat(request, session_id):
    session = get_object_or_404(EEGSession, id=session_id)
    prediction = get_object_or_404(EmotionPrediction, session=session)

    if request.method == "POST":
        message = request.POST.get("message", "").strip()

        if message:
            # Save user message
            ChatMessage.objects.create(
                session=session,
                role="user",
                content=message
            )

            try:
                pred_dict = {
                    "emotion": prediction.emotion,
                    "summary": prediction.summary,
                    "valence": {
                        "label": prediction.valence,
                        "confidence": prediction.valence_conf,
                    },
                    "arousal": {
                        "label": prediction.arousal,
                        "confidence": prediction.arousal_conf,
                    },
                    "dominance": {
                        "label": prediction.dominance,
                        "confidence": prediction.dominance_conf,
                    },
                    "raw_scores": {},
                }

                agent = get_agent()

                result = agent.followup(
                    question=message,
                    session_id=str(session.id),
                    prediction=pred_dict,
                    user_id=str(request.user.id)
                    if request.user.is_authenticated
                    else "anonymous",
                )

                reply = result.get(
                    "explanation",
                    "I could not generate a response."
                )

                ChatMessage.objects.create(
                    session=session,
                    role="agent",
                    content=reply
                )

            except Exception as e:
                ChatMessage.objects.create(
                    session=session,
                    role="agent",
                    content=f"⚠ Agent error: {str(e)}"
                )

        return redirect("chat", session_id=session.id)

    # GET request
    messages = ChatMessage.objects.filter(
        session=session
    ).order_by("created_at")

    return render(request, "eeg_app/chat.html", {
        "session": session,
        "prediction": prediction,
        "messages": messages,
    })



def about(request):
    return render(request, 'eeg_app/about.html')



# ─────────────────────────────────────────────
# AJAX ENDPOINTS (NO AUTH REQUIRED)
# ─────────────────────────────────────────────


@require_POST
def api_chat(request, session_id):

    session = get_object_or_404(EEGSession, id=session_id)
    message = request.POST.get('message', '').strip()

    if not message:
        return HttpResponse("")

    # Save user message
    user_msg = ChatMessage.objects.create(
        session=session,
        role='user',
        content=message
    )

    try:
        pred_obj = session.prediction
        pred_dict = {
            'emotion': pred_obj.emotion,
            'summary': pred_obj.summary,
            'valence': {'label': pred_obj.valence, 'confidence': pred_obj.valence_conf},
            'arousal': {'label': pred_obj.arousal, 'confidence': pred_obj.arousal_conf},
            'dominance': {'label': pred_obj.dominance, 'confidence': pred_obj.dominance_conf},
            'raw_scores': {},
        }

        agent = get_agent()
        result = agent.followup(
            question=message,
            session_id=str(session.id),
            prediction=pred_dict,
            user_id=str(request.user.id) if request.user.is_authenticated else 'anonymous',
        )

        reply = result.get('explanation', 'I could not generate a response.')

        agent_msg = ChatMessage.objects.create(
            session=session,
            role='agent',
            content=reply,
            sources=result.get('sources', []),
            rag_quality=result.get('rag_quality_score', 0),
        )

        # 🔥 RETURN PARTIAL HTML
        return render(request, "eeg_app/partials/chat_pair.html", {
            "user_msg": user_msg,
            "agent_msg": agent_msg,
        })

    except Exception as e:
        return HttpResponse(f"<div class='text-red-400'>Error: {str(e)}</div>")


def api_journal_stats(request):
    """Return journal stats as JSON for chart rendering."""
    user = request.user if request.user.is_authenticated else None
    entries = JournalEntry.objects.filter(user=user) if user else JournalEntry.objects.all()
    emotion_counts = (entries.values('emotion')
                      .annotate(count=Count('emotion'))
                      .order_by('-count'))
    return JsonResponse({
        'total'         : entries.count(),
        'positive'      : entries.filter(valence='High').count(),
        'distress'      : entries.filter(distress=True).count(),
        'emotion_counts': list(emotion_counts),
    })


def api_delete_session(request, session_id):
    session = get_object_or_404(EEGSession, id=session_id)
    session.delete()
    return JsonResponse({'ok': True})

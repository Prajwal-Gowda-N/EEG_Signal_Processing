"""
Advanced Emotion RAG Agent  —  LangGraph
==========================================
Features implemented:
  ✅ Hybrid search        (FAISS semantic + BM25 keyword combined)
  ✅ Query rewriting      (vague question → better search query)
  ✅ Reranking            (score chunks by relevance, keep best)
  ✅ Self-reflection      (agent checks own answer, rewrites if poor)
  ✅ Multi-turn memory    (full session history remembered)
  ✅ Auto follow-ups      (agent generates 3 suggested questions)
  ✅ Periodic summary     (every 5 turns, summarize conversation)
  ✅ Mood trend context   (journal history injected into every prompt)
  ✅ Distress detection   (flag + gentle referral message)
  ✅ Meditation recs      (tailored breathing/meditation exercises)
  ✅ Emotion journal      (every session logged automatically)

LangGraph Node Flow
────────────────────
  parse_prediction
       ↓
  inject_journal_context      ← mood history from EmotionJournal
       ↓
  rewrite_query               ← LLM rewrites search query
       ↓
  hybrid_retrieve             ← FAISS + BM25 combined
       ↓
  rerank_chunks               ← score by relevance, keep top K
       ↓
  generate_explanation        ← LLM generates patient explanation
       ↓
  self_reflect                ← LLM scores own answer (retry if poor)
       ↓
  add_recommendations         ← meditation / referral if needed
       ↓
  generate_followup_questions ← 3 suggested questions for user
       ↓
  maybe_summarize             ← summarize if turn count ≥ 5
       ↓
  format_response             ← clean dict for Django

Install:
  pip install langgraph langchain langchain-community langchain-anthropic
              faiss-cpu sentence-transformers rank-bm25

LLM options (set env vars):
  Anthropic : ANTHROPIC_API_KEY=sk-...        (recommended)
  Ollama    : LLM_BACKEND=ollama  OLLAMA_MODEL=llama3.2
"""

import os
import json
import uuid
import pickle
import time
from typing import TypedDict, List, Optional, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage


from .emotional_journel import EmotionJournal   # ✅ WORKS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

VS_DIR         = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/EEG/rag/vectorstore/emotion"

EMBED_MODEL    = "all-MiniLM-L6-v2"

TOP_K_RETRIEVE = 8    # retrieve this many chunks before reranking
TOP_K_RERANK   = 3    # keep this many after reranking
REFLECT_THRESHOLD = 4  # quality score below this → rewrite answer (0-10)
SUMMARIZE_EVERY   = 5  # summarize conversation every N turns

LLM_BACKEND   = os.getenv("LLM_BACKEND",   "gemini")
GEMINI_MODEL  = os.getenv("GEMINI_MODEL",  "gemini-2.0-flash")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama3.2")

# ─────────────────────────────────────────────
# LLM LOADER
# ─────────────────────────────────────────────

def load_llm():
    """
    Load LLM backend. Priority:
      1. Gemini Flash 2.0  (default — fast, generous free tier)
      2. Ollama            (local fallback)
      3. HuggingFace       (offline fallback)
    """
    backend = LLM_BACKEND.lower()

    if backend == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set.\n"
                    "Get free key at: https://aistudio.google.com/app/apikey\n"
                    "Then: set GOOGLE_API_KEY=your_key_here")
            llm = ChatGoogleGenerativeAI(
                model          = GEMINI_MODEL,   # gemini-2.0-flash
                temperature    = 0.3,
                max_output_tokens = 1024,
                google_api_key = api_key,
            )
            # Quick test
            llm.invoke([HumanMessage(content="hi")])
            print(f"[LLM] ✓ Google Gemini ({GEMINI_MODEL})")
            return llm
        except Exception as e:
            print(f"[LLM] Gemini failed: {e}")
            print(f"[LLM] Falling back to Ollama ...")
            backend = "ollama"

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
            llm.invoke([HumanMessage(content="hi")])
            print(f"[LLM] ✓ Ollama ({OLLAMA_MODEL})")
            return llm
        except Exception as e:
            print(f"[LLM] Ollama failed: {e}")
            print(f"[LLM] Falling back to HuggingFace ...")
            backend = "huggingface"

    if backend == "huggingface":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        pipe = pipeline("text-generation", model="google/flan-t5-base",
                        max_new_tokens=512)
        print("[LLM] ✓ HuggingFace flan-t5-base (local, no API key needed)")
        return HuggingFacePipeline(pipeline=pipe)

    raise RuntimeError("No LLM backend available.")

# ─────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────
    prediction      : dict
    session_id      : str
    user_id         : str
    user_question   : str          # "" for initial call, filled for follow-ups
    turn_count      : int

    # ── Intermediate ───────────────────────────
    parsed          : dict
    journal_context : str          # mood history summary from journal
    rewritten_query : str          # improved search query
    retrieved_docs  : List[dict]   # raw retrieved chunks with scores
    reranked_docs   : List[dict]   # top K after reranking
    context_text    : str          # final context fed to LLM

    # ── Generation ─────────────────────────────
    explanation     : str
    reflection_score: int          # 0-10 quality score
    reflection_note : str          # what was wrong (if score < threshold)
    rewrite_count   : int          # how many times we've retried

    # ── Enrichment ─────────────────────────────
    meditation_rec  : str          # tailored meditation exercise
    distress_flag   : bool
    referral_message: str
    followup_questions: List[str]  # 3 suggested questions for user
    conversation_summary: str      # periodic summary

    # ── Output ─────────────────────────────────
    action_steps    : List[str]
    sources         : List[str]
    chat_history    : List[dict]
    final_response  : dict


# ─────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS (semantic) + BM25 (keyword) search.
    Results from both are merged and deduplicated.

    Score fusion: (semantic_score * 0.6) + (bm25_score * 0.4)
    """

    def __init__(self, vs_dir: str, embeddings):
        self.faiss = FAISS.load_local(
            vs_dir, embeddings,
            allow_dangerous_deserialization=True)

        bm25_path = os.path.join(vs_dir, 'bm25.pkl')
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25  = data['bm25']
            self.texts = data['texts']
            self.metas = data['metas']
            self.has_bm25 = True
        else:
            self.has_bm25 = False
            print("[Retriever] BM25 index not found — using FAISS only")

    def retrieve(self, query: str, k: int = 8) -> List[dict]:
        """
        Hybrid search: FAISS + BM25 → merged results.
        Returns list of dicts with text, metadata, score.
        """
        results = {}

        # ── FAISS semantic search ──────────────
        faiss_results = self.faiss.similarity_search_with_score(query, k=k)
        for doc, score in faiss_results:
            key = doc.page_content[:100]
            results[key] = {
                'text'    : doc.page_content,
                'metadata': doc.metadata,
                'faiss_score': float(score),
                'bm25_score' : 0.0,
            }

        # ── BM25 keyword search ────────────────
        if self.has_bm25:
            tokens     = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokens)
            top_indices = bm25_scores.argsort()[-k:][::-1]

            max_bm25 = bm25_scores[top_indices[0]] + 1e-8  # normalize

            for idx in top_indices:
                text = self.texts[idx]
                key  = text[:100]
                norm_score = float(bm25_scores[idx]) / max_bm25

                if key in results:
                    results[key]['bm25_score'] = norm_score
                else:
                    results[key] = {
                        'text'      : text,
                        'metadata'  : self.metas[idx],
                        'faiss_score': 0.0,
                        'bm25_score' : norm_score,
                    }

        # ── Fuse scores ────────────────────────
        for key in results:
            r = results[key]
            r['hybrid_score'] = (r['faiss_score'] * 0.6 +
                                 r['bm25_score']  * 0.4)

        sorted_results = sorted(results.values(),
                                key=lambda x: x['hybrid_score'],
                                reverse=True)
        return sorted_results[:k]


# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────

class EmotionAgentNodes:

    def __init__(self, retriever: HybridRetriever, llm, journal: EmotionJournal):
        self.retriever = retriever
        self.llm       = llm
        self.journal   = journal

    def _call_llm(self, prompt: str) -> str:
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, 'content') else str(resp)

    # ── Node 1 ────────────────────────────────
    def parse_prediction(self, state: AgentState) -> dict:
        """Parse raw emotion_inference output into clean fields."""
        pred = state['prediction']
        v    = pred.get('valence',   {})
        a    = pred.get('arousal',   {})
        d    = pred.get('dominance', {})

        parsed = {
            'emotion'        : pred.get('emotion',  'Unknown'),
            'summary'        : pred.get('summary',  ''),
            'valence_label'  : v.get('label',       'Unknown'),
            'valence_conf'   : v.get('confidence',   0.0),
            'arousal_label'  : a.get('label',        'Unknown'),
            'arousal_conf'   : a.get('confidence',   0.0),
            'dominance_label': d.get('label',        'Unknown'),
            'dominance_conf' : d.get('confidence',   0.0),
            'raw_scores'     : pred.get('raw_scores', {}),
        }
        return {
            'parsed'      : parsed,
            'turn_count'  : state.get('turn_count', 0) + 1,
            'rewrite_count': 0,
        }

    # ── Node 2 ────────────────────────────────
    def inject_journal_context(self, state: AgentState) -> dict:
        """Load mood history from journal and inject as context."""
        user_id = state.get('user_id', 'default')
        self.journal.user_id = user_id

        journal_context = self.journal.summarize_for_agent()
        distress_check  = self.journal.check_distress_trend()

        return {
            'journal_context': journal_context,
            'distress_flag'  : distress_check['flagged'],
            'referral_message': distress_check['message'],
        }

    # ── Node 3 ────────────────────────────────
    def rewrite_query(self, state: AgentState) -> dict:
        """
        LLM rewrites the search query to be more specific
        before hitting the vector store.
        """
        parsed   = state['parsed']
        question = state.get('user_question', '')

        if question:
            # Follow-up mode: rewrite user question
            prompt = f"""You are a search query optimizer for a neuroscience knowledge base.

The user's original question: "{question}"

Context: This is about EEG emotion analysis.
Current emotional state: {parsed['emotion']} 
(Valence: {parsed['valence_label']}, Arousal: {parsed['arousal_label']})

Rewrite the question into an optimal search query for retrieving 
relevant neuroscience/psychology content. 
Return ONLY the rewritten query, nothing else. Max 15 words."""
        else:
            # Initial explanation: build query from prediction
            prompt = f"""You are a search query optimizer for a neuroscience knowledge base.

Generate an optimal search query to retrieve information about:
- Detected emotion: {parsed['emotion']}
- Valence: {parsed['valence_label']}
- Arousal: {parsed['arousal_label']}
- Dominance: {parsed['dominance_label']}

The query should find relevant EEG neuroscience, emotion psychology,
and brain activity research. 
Return ONLY the search query. Max 15 words."""

        rewritten = self._call_llm(prompt).strip().strip('"').strip("'")
        print(f"  [QueryRewrite] '{question or parsed['emotion']}' → '{rewritten}'")
        return {'rewritten_query': rewritten}

    # ── Node 4 ────────────────────────────────
    def hybrid_retrieve(self, state: AgentState) -> dict:
        """Hybrid FAISS + BM25 retrieval."""
        query = state['rewritten_query']
        docs  = self.retriever.retrieve(query, k=TOP_K_RETRIEVE)
        print(f"  [Retrieve] {len(docs)} chunks retrieved (hybrid)")
        return {'retrieved_docs': docs}

    # ── Node 5 ────────────────────────────────
    def rerank_chunks(self, state: AgentState) -> dict:
        """
        LLM-based reranking: score each chunk 0-10 for relevance.
        Keep top TOP_K_RERANK chunks.
        """
        query = state['rewritten_query']
        docs  = state['retrieved_docs']

        scored = []
        for doc in docs[:6]:   # score top 6 candidates max
            prompt = f"""Rate how relevant this text is to the query on a scale of 0-10.
Query: "{query}"
Text: "{doc['text'][:300]}"
Return ONLY a number 0-10."""
            try:
                score_str = self._call_llm(prompt).strip()
                score     = float(''.join(c for c in score_str if c.isdigit() or c=='.'))
                score     = min(10.0, max(0.0, score))
            except Exception:
                score = doc['hybrid_score'] * 10

            scored.append({**doc, 'rerank_score': score})
            time.sleep(0.1)

        reranked = sorted(scored, key=lambda x: x['rerank_score'], reverse=True)
        top      = reranked[:TOP_K_RERANK]

        context = "\n\n---\n\n".join([
            f"[Source: {d['metadata'].get('source','Unknown')} | "
            f"Score: {d['rerank_score']:.1f}/10]\n{d['text']}"
            for d in top
        ])
        sources  = list({d['metadata'].get('source','Unknown') for d in top})

        print(f"  [Rerank] Top scores: "
              f"{[round(d['rerank_score'],1) for d in top]}")
        return {
            'reranked_docs': top,
            'context_text' : context,
            'sources'      : sources,
        }

    # ── Node 6 ────────────────────────────────
    def generate_explanation(self, state: AgentState) -> dict:
        """
        Core LLM generation.
        Builds patient-friendly explanation + action steps.
        Uses journal context for personalization.
        """
        parsed          = state['parsed']
        context         = state['context_text']
        journal_ctx     = state.get('journal_context', '')
        question        = state.get('user_question', '')
        history         = state.get('chat_history', [])
        conv_summary    = state.get('conversation_summary', '')
        rewrite_count   = state.get('rewrite_count', 0)
        reflection_note = state.get('reflection_note', '')

        # Build history text (last 6 messages)
        history_text = ""
        if conv_summary:
            history_text = f"Conversation summary so far:\n{conv_summary}\n\n"
        if history:
            recent = history[-6:]
            history_text += "\n".join([
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
                for m in recent
            ])

        # Add correction note if rewriting
        correction = ""
        if rewrite_count > 0 and reflection_note:
            correction = f"\nIMPORTANT: Previous answer was inadequate ({reflection_note}). Fix this.\n"

        if question:
            prompt = f"""You are a warm, friendly medical AI assistant specializing in EEG emotion analysis.

User's emotional state from EEG:
- Emotion: {parsed['emotion']}
- Valence: {parsed['valence_label']} | Arousal: {parsed['arousal_label']} | Dominance: {parsed['dominance_label']}

User's history: {journal_ctx}

{history_text}

Relevant neuroscience knowledge:
{context}
{correction}
User's question: "{question}"

Answer warmly in plain language (no jargon). Be specific and helpful.
Keep your answer to 3-5 sentences. End with one empathetic sentence."""

        else:
            prompt = f"""You are a warm, friendly medical AI assistant explaining EEG emotion results.

EEG analysis results:
- Emotion detected: {parsed['emotion']}
- Valence (pleasantness): {parsed['valence_label']} ({parsed['valence_conf']:.0%})
- Arousal (energy level): {parsed['arousal_label']} ({parsed['arousal_conf']:.0%})
- Dominance (control):    {parsed['dominance_label']} ({parsed['dominance_conf']:.0%})

User's emotional history: {journal_ctx}

Relevant neuroscience knowledge:
{context}
{correction}

Write a warm, patient-friendly explanation in exactly 3 sections:

WHAT THIS MEANS:
(2-3 sentences. What does this emotional state mean? 
Which brain regions are involved? Use simple words.)

WHAT YOU MIGHT BE FEELING:
(1-2 sentences describing the felt experience of this emotion.)

WHAT YOU CAN DO RIGHT NOW:
(List 3 specific, practical, numbered steps. 
Match steps to the emotional state — calming for high arousal, 
energizing for low arousal, etc.)

Tone: warm, reassuring, never alarming."""

        explanation = self._call_llm(prompt)

        # Extract action steps
        action_steps = []
        for marker in ["WHAT YOU CAN DO RIGHT NOW:", "WHAT YOU CAN DO:"]:
            if marker in explanation:
                section = explanation.split(marker)[-1].strip()
                lines   = [l.strip() for l in section.split('\n') if l.strip()]
                action_steps = [l.lstrip('•-123456789. ') for l in lines[:4] if l]
                break

        # Update chat history
        new_history = list(history)
        if question:
            new_history.append({'role': 'user',      'content': question})
            new_history.append({'role': 'assistant',  'content': explanation})
        else:
            new_history.append({'role': 'assistant',  'content': explanation})

        return {
            'explanation' : explanation,
            'action_steps': action_steps,
            'chat_history': new_history,
        }

    # ── Node 7 ────────────────────────────────
    def self_reflect(self, state: AgentState) -> dict:
        """
        Agent scores its own answer 0-10.
        If score < REFLECT_THRESHOLD and retries < 2 → flag for rewrite.
        """
        explanation   = state['explanation']
        question      = state.get('user_question', '')
        parsed        = state['parsed']
        rewrite_count = state.get('rewrite_count', 0)

        prompt = f"""You are a quality reviewer for medical AI explanations.

Rate this explanation on a scale of 0-10 based on:
- Clarity (is it easy to understand for a patient?)
- Accuracy (does it correctly reflect the emotional state?)
- Helpfulness (does it give actionable guidance?)
- Empathy (is it warm and reassuring?)

Emotional state: {parsed['emotion']}
{"Question: " + question if question else "Initial explanation"}

Explanation to rate:
"{explanation[:600]}"

Respond in EXACTLY this format:
SCORE: [0-10]
REASON: [one sentence explaining the score]"""

        try:
            response = self._call_llm(prompt)
            lines    = response.strip().split('\n')
            score    = 7   # default if parsing fails
            reason   = ""
            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = int(''.join(c for c in line.split(':')[1] if c.isdigit()))
                        score = min(10, max(0, score))
                    except Exception:
                        score = 7
                if line.startswith("REASON:"):
                    reason = line.split(':', 1)[-1].strip()

            print(f"  [SelfReflect] Score={score}/10  '{reason[:60]}'")
        except Exception:
            score  = 7
            reason = ""

        should_rewrite = (score < REFLECT_THRESHOLD and rewrite_count < 2)

        return {
            'reflection_score': score,
            'reflection_note' : reason if should_rewrite else "",
            'rewrite_count'   : rewrite_count + (1 if should_rewrite else 0),
        }

    # ── Node 8 ────────────────────────────────
    def add_recommendations(self, state: AgentState) -> dict:
        """
        Add tailored meditation/breathing recommendation.
        Add therapist referral if distress flagged.
        Log session to journal.
        """
        parsed   = state['parsed']
        valence  = parsed['valence_label']
        arousal  = parsed['arousal_label']

        # ── Meditation recommendation ──────────
        meditation_map = {
            ('Low',  'High'): {
                'name'       : '4-7-8 Breathing',
                'description': ('Inhale for 4 counts, hold for 7, '
                                'exhale for 8. Repeat 4 times. '
                                'This activates your parasympathetic '
                                'nervous system and reduces anxiety.'),
                'duration'   : '5 minutes',
            },
            ('Low',  'Low'): {
                'name'       : 'Energizing Breath (Kapalabhati)',
                'description': ('Take a deep inhale, then do 20 short '
                                'sharp exhales through your nose. '
                                'This increases alertness and positive energy.'),
                'duration'   : '3 minutes',
            },
            ('High', 'High'): {
                'name'       : 'Body Scan Meditation',
                'description': ('Lie down and mentally scan from toes to '
                                'head, releasing tension. Focus on warmth '
                                'and heaviness in each body part.'),
                'duration'   : '10 minutes',
            },
            ('High', 'Low'): {
                'name'       : 'Gratitude Breathing',
                'description': ('Take 5 slow deep breaths. With each inhale '
                                'think of something you are grateful for. '
                                'This sustains positive emotional states.'),
                'duration'   : '5 minutes',
            },
        }

        med = meditation_map.get(
            (valence, arousal),
            {
                'name'       : 'Mindful Breathing',
                'description': 'Take 10 slow, deep breaths. Focus entirely '
                               'on the sensation of breathing.',
                'duration'   : '5 minutes',
            }
        )
        meditation_rec = (f"🧘 {med['name']} ({med['duration']}): "
                          f"{med['description']}")

        # ── Referral message ───────────────────
        referral = ""
        if state.get('distress_flag'):
            referral = (
                "💙 Your recent EEG patterns suggest you may be experiencing "
                "ongoing emotional distress. Consider speaking with a mental "
                "health professional — it's a sign of strength to seek support. "
                "Resources: iCall (9152987821) · Vandrevala Foundation (1860-2662-345)"
            )

        # ── Log to journal ─────────────────────
        self.journal.log(
            prediction  = state['prediction'],
            explanation = state['explanation'],
            session_id  = state['session_id'],
        )

        return {
            'meditation_rec'  : meditation_rec,
            'referral_message': referral,
        }

    # ── Node 9 ────────────────────────────────
    def generate_followup_questions(self, state: AgentState) -> dict:
        """
        Generate 3 natural follow-up questions the user might want to ask.
        Shown as clickable suggestions in the Django UI.
        """
        parsed = state['parsed']
        prompt = f"""A patient just received an EEG emotion analysis result:
Emotion: {parsed['emotion']}
Valence: {parsed['valence_label']} | Arousal: {parsed['arousal_label']}

Generate exactly 3 short, natural follow-up questions the patient might ask.
Make them conversational and specific to this emotional state.
Return ONLY the 3 questions, one per line, no numbering, no bullets."""

        try:
            response = self._call_llm(prompt)
            questions = [q.strip().strip('•-123. ')
                         for q in response.strip().split('\n')
                         if q.strip() and '?' in q][:3]
        except Exception:
            questions = [
                f"Why does my brain show {parsed['arousal_label'].lower()} arousal?",
                "How can I improve my emotional state?",
                "Is this emotional pattern normal?",
            ]

        return {'followup_questions': questions}

    # ── Node 10 ───────────────────────────────
    def maybe_summarize(self, state: AgentState) -> dict:
        """
        Every SUMMARIZE_EVERY turns, summarize conversation so far.
        Keeps the context window lean for long sessions.
        """
        turn_count = state.get('turn_count', 1)
        history    = state.get('chat_history', [])

        if turn_count % SUMMARIZE_EVERY != 0 or len(history) < 4:
            return {}

        history_text = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}"
            for m in history[-10:]
        ])
        prompt = f"""Summarize this EEG emotion counseling conversation in 3-4 sentences.
Focus on: key emotional insights, what was explained, and what was recommended.

Conversation:
{history_text}

Return ONLY the summary."""

        try:
            summary = self._call_llm(prompt).strip()
        except Exception:
            summary = state.get('conversation_summary', '')

        print(f"  [Summarize] Turn {turn_count} — conversation summarized")
        return {'conversation_summary': summary}

    # ── Node 11 ───────────────────────────────
    def format_response(self, state: AgentState) -> dict:
        """Package everything into final clean dict for Django."""
        parsed = state['parsed']
        response = {
            'session_id'          : state.get('session_id', str(uuid.uuid4())),
            'user_id'             : state.get('user_id', 'default'),
            'timestamp'           : datetime.now().isoformat(),
            'service'             : 'emotion',
            'turn_count'          : state.get('turn_count', 1),

            # Prediction
            'prediction': {
                'emotion'   : parsed['emotion'],
                'summary'   : parsed['summary'],
                'valence'   : parsed['valence_label'],
                'arousal'   : parsed['arousal_label'],
                'dominance' : parsed['dominance_label'],
                'confidence': {
                    'valence'  : round(parsed['valence_conf'],   2),
                    'arousal'  : round(parsed['arousal_conf'],   2),
                    'dominance': round(parsed['dominance_conf'], 2),
                }
            },

            # Agent outputs
            'explanation'         : state.get('explanation',          ''),
            'action_steps'        : state.get('action_steps',         []),
            'meditation'          : state.get('meditation_rec',        ''),
            'referral'            : state.get('referral_message',      ''),
            'distress_flag'       : state.get('distress_flag',        False),
            'followup_questions'  : state.get('followup_questions',    []),
            'conversation_summary': state.get('conversation_summary',  ''),

            # RAG metadata
            'sources'             : state.get('sources',               []),
            'rag_quality_score'   : state.get('reflection_score',       0),
            'query_rewritten_to'  : state.get('rewritten_query',        ''),

            # Session data
            'chat_history'        : state.get('chat_history',           []),
            'journal_context'     : state.get('journal_context',        ''),
        }
        return {'final_response': response}


# ─────────────────────────────────────────────
# ROUTING — self-reflection retry
# ─────────────────────────────────────────────

def should_rewrite(state: AgentState) -> str:
    """
    Conditional edge: if reflection score too low AND retries < 2
    → loop back to generate_explanation.
    Otherwise → proceed.
    """
    score  = state.get('reflection_score', 10)
    count  = state.get('rewrite_count',     0)

    if score < REFLECT_THRESHOLD and count <= 2:
        print(f"  [Route] Score {score} < {REFLECT_THRESHOLD} → rewriting "
              f"(attempt {count})")
        return "rewrite"
    return "proceed"


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_emotion_graph(nodes: EmotionAgentNodes):
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_prediction",          nodes.parse_prediction)
    workflow.add_node("inject_journal_context",    nodes.inject_journal_context)
    workflow.add_node("rewrite_query",             nodes.rewrite_query)
    workflow.add_node("hybrid_retrieve",           nodes.hybrid_retrieve)
    workflow.add_node("rerank_chunks",             nodes.rerank_chunks)
    workflow.add_node("generate_explanation",      nodes.generate_explanation)
    workflow.add_node("self_reflect",              nodes.self_reflect)
    workflow.add_node("add_recommendations",       nodes.add_recommendations)
    workflow.add_node("generate_followup_questions",nodes.generate_followup_questions)
    workflow.add_node("maybe_summarize",           nodes.maybe_summarize)
    workflow.add_node("format_response",           nodes.format_response)

    # ── Linear edges ──────────────────────────
    workflow.set_entry_point("parse_prediction")
    workflow.add_edge("parse_prediction",         "inject_journal_context")
    workflow.add_edge("inject_journal_context",   "rewrite_query")
    workflow.add_edge("rewrite_query",            "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve",          "rerank_chunks")
    workflow.add_edge("rerank_chunks",            "generate_explanation")
    workflow.add_edge("generate_explanation",     "self_reflect")

    # ── Conditional edge: reflect → rewrite or proceed ──
    workflow.add_conditional_edges(
        "self_reflect",
        should_rewrite,
        {
            "rewrite" : "generate_explanation",   # loop back
            "proceed" : "add_recommendations",    # move forward
        }
    )

    workflow.add_edge("add_recommendations",          "generate_followup_questions")
    workflow.add_edge("generate_followup_questions",  "maybe_summarize")
    workflow.add_edge("maybe_summarize",              "format_response")
    workflow.add_edge("format_response",              END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# PUBLIC AGENT CLASS
# ─────────────────────────────────────────────

class EmotionAgent:
    """
    Main entry point for Django to use.

    Usage:
        agent = EmotionAgent()

        # Initial explanation
        result = agent.explain(prediction, user_id="user123")

        # Follow-up question
        answer = agent.followup("Why am I anxious?",
                                session_id=result['session_id'],
                                prediction=prediction,
                                user_id="user123")

        # Get journal stats for dashboard
        stats = agent.get_journal_stats(user_id="user123")
    """

    def __init__(self):
        print("[EmotionAgent] Initializing ...")

        self.llm = load_llm()

        embeddings = HuggingFaceEmbeddings(
            model_name    = EMBED_MODEL,
            model_kwargs  = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True},
        )

        if not os.path.exists(VS_DIR):
            raise FileNotFoundError(
                f"Vectorstore not found: {VS_DIR}\n"
                f"Run: python rag_builder.py first")

        retriever = HybridRetriever(VS_DIR, embeddings)
        journal   = EmotionJournal()
        nodes     = EmotionAgentNodes(retriever, self.llm, journal)

        self.graph   = build_emotion_graph(nodes)
        self.journal = journal
        print("[EmotionAgent] Ready  ✓")

    def _run(self, state: dict, session_id: str) -> dict:
        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(state, config=config)
        return result['final_response']

    def explain(self, prediction: dict, user_id: str = "default",
                session_id: str = None) -> dict:
        """Initial explanation for a new EEG result."""
        session_id = session_id or str(uuid.uuid4())
        state = {
            'prediction'          : prediction,
            'session_id'          : session_id,
            'user_id'             : user_id,
            'user_question'       : '',
            'turn_count'          : 0,
            'chat_history'        : [],
            'conversation_summary': '',
            'rewrite_count'       : 0,
            'reflection_score'    : 10,
            'reflection_note'     : '',
        }
        return self._run(state, session_id)

    def followup(
    self,
    question: str,
    session_id: str,
    prediction: dict,
    user_id: str = "default"
) -> dict:

        config = {"configurable": {"thread_id": session_id}}

    # 🔹 Load previous graph state safely
        try:
            existing = self.graph.get_state(config)

            if existing and hasattr(existing, "values"):
                state = dict(existing.values)
            else:
                 state = {}

        except Exception:
            state = {}

    # 🔥 ALWAYS re-inject required core fields
        state["prediction"] = prediction
        state["session_id"] = session_id
        state["user_id"] = user_id
        state["user_question"] = question

    # 🔹 Preserve memory fields safely
        state["turn_count"] = state.get("turn_count", 0)
        state["chat_history"] = state.get("chat_history", [])
        state["conversation_summary"] = state.get("conversation_summary", "")
        state["rewrite_count"] = 0
        state["reflection_score"] = 10
        state["reflection_note"] = ""

        return self._run(state, session_id)

    def get_journal_stats(self, user_id: str = "default") -> dict:
        """Return mood trends + distress status for Django dashboard."""
        self.journal.user_id      = user_id
        self.journal.journal_path = f"rag/journal/journal_{user_id}.json"
        self.journal.records      = self.journal._load()
        return self.journal.get_stats()

    def get_journal_entries(self, user_id: str = "default",
                            n: int = 20) -> list:
        """Return recent journal entries for Django journal page."""
        self.journal.user_id = user_id
        self.journal.records = self.journal._load()
        return self.journal.get_recent(n=n)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Advanced Emotion Agent  —  Quick Test")
    print("=" * 60)

    agent = EmotionAgent()

    dummy_pred = {
        'valence'   : {'label': 'Low',  'confidence': 0.81},
        'arousal'   : {'label': 'High', 'confidence': 0.79},
        'dominance' : {'label': 'Low',  'confidence': 0.74},
        'emotion'   : 'Anxious / Nervous',
        'summary'   : 'Low Valence · High Arousal · Low Dominance',
        'raw_scores': {'valence': 0, 'arousal': 1, 'dominance': 0},
    }

    # ── Initial explanation ────────────────────
    print("\n  [1] Initial explanation ...")
    result     = agent.explain(dummy_pred, user_id="test_user")
    session_id = result['session_id']

    print(f"\n  Emotion     : {result['prediction']['emotion']}")
    print(f"  RAG Quality : {result['rag_quality_score']}/10")
    print(f"  Query used  : {result['query_rewritten_to']}")
    print(f"\n  Explanation :\n  {result['explanation'][:400]}...")
    print(f"\n  Action Steps:")
    for i, s in enumerate(result['action_steps'], 1):
        print(f"    {i}. {s}")
    print(f"\n  Meditation  : {result['meditation']}")
    print(f"\n  Suggested follow-ups:")
    for q in result['followup_questions']:
        print(f"    • {q}")
    print(f"\n  Sources     : {result['sources']}")
    print(f"  Distress    : {result['distress_flag']}")

    # ── Follow-up question ─────────────────────
    print(f"\n  [2] Follow-up question ...")
    followup = agent.followup(
        result['followup_questions'][0],
        session_id  = session_id,
        prediction  = dummy_pred,
        user_id     = "test_user",
    )
    print(f"  Q: {result['followup_questions'][0]}")
    print(f"  A: {followup['explanation'][:300]}...")

    # ── Journal stats ──────────────────────────
    print(f"\n  [3] Journal stats ...")
    stats = agent.get_journal_stats("test_user")
    print(f"  Stats: {json.dumps(stats, indent=2)[:400]}")

    print(f"\n{'='*60}")
    print(f"  All features working. Ready for Django.")
    print(f"{'='*60}")
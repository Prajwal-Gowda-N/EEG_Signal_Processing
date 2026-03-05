"""
RAG Knowledge Base Builder — FIXED VERSION (arXiv + Wikipedia + PDFs)
==============================================================
Builds FAISS vectorstore + BM25 hybrid search for EEG emotion recognition.
"""

import os
import json
import time
import pickle
import requests
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RAG_DIR     = "rag"
VS_DIR      = os.path.join(RAG_DIR, "vectorstore", "emotion")
BOOKS_DIR   = os.path.join(RAG_DIR, "books")

# Create directories
for d in [RAG_DIR, VS_DIR, BOOKS_DIR]:
    os.makedirs(d, exist_ok=True)

# RAG Parameters
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100
EMBED_MODEL    = "all-MiniLM-L6-v2"

# Knowledge Sources
EMOTION_ARXIV = [
    "1901.05548",  # EEG emotion recognition review
    "2011.10591",  # EEG emotion deep learning
    "1809.09407",  # Affective computing EEG survey
    "2206.09108",  # EEGNet
    "1708.01073",  # DEAP dataset
    "2107.09598",  # Transformer EEG emotion
    "1911.10604",  # Multi-modal emotion recognition
    "2301.02284",  # Recent EEG emotion review
    "2203.14415",  # Deep learning affective computing
]

EMOTION_WIKI = [
    "Electroencephalography", "Emotion_recognition", "Affective_computing",
    "Valence_(psychology)", "Arousal", "Dominance_(psychology)",
    "Amygdala", "Limbic_system", "Prefrontal_cortex",
    "Alpha_wave", "Beta_wave", "Theta_wave", "Gamma_wave", "Delta_wave",
    "Anxiety", "Stress_(biology)", "Mindfulness", "Meditation",
    "Cognitive_behavioral_therapy"
]

# ─────────────────────────────────────────────
# HTTP HEADERS (Wikipedia compliance)
# ─────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'EEG-RAG-Builder/2.0 (contact@prajwal.ai; Bengaluru)',
    'Accept': 'application/json',
    'From': 'prajwal@brain-signaling.in'
}

# ─────────────────────────────────────────────
# DATA FETCHERS (FIXED)
# ─────────────────────────────────────────────

def fetch_arxiv(paper_id: str) -> dict | None:
    """Fetch arXiv paper abstract + metadata."""
    api = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        resp = requests.get(api, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(resp.text)
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            return None

        title = entry.find('atom:title', ns).text.strip()
        abstract = entry.find('atom:summary', ns).text.strip()
        authors = [a.find('atom:name', ns).text.strip() 
                  for a in entry.findall('atom:author', ns)][:3]

        text = f"TITLE: {title}\nAUTHORS: {', '.join(authors)}\n\nABSTRACT:\n{abstract}"
        
        return {
            'text': text,
            'source': f"arXiv:{paper_id}",
            'title': title,
            'url': f"https://arxiv.org/abs/{paper_id}",
            'type': 'arxiv'
        }
    except Exception as e:
        print(f"    [WARN] arXiv:{paper_id}: {str(e)[:50]}")
        return None

def fetch_wikipedia_batch(topics: list) -> list[dict]:
    """Batch fetch Wikipedia articles (rate-limit safe)."""
    results = []
    api = "https://en.wikipedia.org/w/api.php"
    
    # Process in batches of 20 (Wikipedia max)
    for i in range(0, len(topics), 20):
        batch = topics[i:i+20]
        titles_param = '|'.join([t.replace(' ', '_') for t in batch])
        
        params = {
            'action': 'query',
            'prop': 'extracts',
            'explaintext': True,
            'titles': titles_param,
            'format': 'json',
            'redirects': 1,
            'exintro': True,
            'exlimit': 'max',
            'exsectionformat': 'plain'
        }
        
        try:
            resp = requests.get(api, headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            
            if 'query' in data and 'pages' in data['query']:
                for page_id, page in data['query']['pages'].items():
                    if 'missing' not in page and 'extract' in page:
                        extract = page['extract'][:5000]  # Truncate long extracts
                        if len(extract.strip()) > 150:
                            title = page.get('title', 'Unknown')
                            results.append({
                                'text': f"Wikipedia: {title}\n\n{extract}",
                                'source': f"Wikipedia:{title}",
                                'title': title,
                                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                'type': 'wikipedia'
                            })
                            
        except Exception as e:
            print(f"    [BATCH] Error fetching {batch[:2]}: {str(e)[:50]}")
        
        # Rate limiting - 2 seconds between batches
        if i + 20 < len(topics):
            print("    [WAIT] Rate limiting...")
            time.sleep(2)
    
    return results

def load_pdfs(books_dir: str) -> list:
    """Load all PDFs from books directory."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("  [WARN] `pip install pypdf` required for PDF support")
        return []

    docs = []
    pdf_files = [f for f in os.listdir(books_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"  [INFO] No PDFs found in {books_dir} — add neuroscience books here!")
        return docs

    for fname in pdf_files:
        filepath = os.path.join(books_dir, fname)
        try:
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            
            for page in pages:
                page.metadata.update({
                    'source': fname,
                    'page': page.metadata.get('page', 0),
                    'type': 'pdf_book'
                })
            docs.extend(pages)
            print(f"    [PDF] {fname}: {len(pages)} pages loaded ✓")
            
        except Exception as e:
            print(f"    [WARN] Failed to load {fname}: {str(e)[:50]}")
    
    return docs

# ─────────────────────────────────────────────
# MAIN RAG BUILDER
# ─────────────────────────────────────────────

def build():
    print("=" * 70)
    print(" 🧠 EEG EMOTION RAG BUILDER — PRODUCTION READY")
    print("=" * 70)
    print(f"  📊 Embed model:  {EMBED_MODEL}")
    print(f"  🔪 Chunk size:   {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"  💾 Output dir:   {VS_DIR}")
    print("=" * 70)

    # Import LangChain components (modern imports)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi

    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    print("\n" + "="*70)
    raw_docs = []

    # 1️⃣ arXiv Papers
    print(" 📚 1. Fetching arXiv papers...")
    arxiv_count = 0
    for pid in EMOTION_ARXIV:
        print(f"     arXiv:{pid:<10} ", end="", flush=True)
        doc_data = fetch_arxiv(pid)
        if doc_data:
            raw_docs.append(Document(
                page_content=doc_data['text'],
                metadata={
                    'source': doc_data['source'],
                    'title': doc_data['title'],
                    'url': doc_data['url'],
                    'type': 'arxiv',
                    'domain': 'emotion'
                }
            ))
            arxiv_count += 1
            print("✓")
        else:
            print("✗")
        time.sleep(1)  # Rate limit

    # 2️⃣ Wikipedia Articles
    print("\n 🌐 2. Fetching Wikipedia articles (batched)...")
    wiki_docs = fetch_wikipedia_batch(EMOTION_WIKI)
    wiki_count = len(wiki_docs)
    for doc_data in wiki_docs:
        raw_docs.append(Document(
            page_content=doc_data['text'],
            metadata={
                'source': doc_data['source'],
                'title': doc_data['title'],
                'url': doc_data['url'],
                'type': 'wikipedia',
                'domain': 'emotion'
            }
        ))
    print(f"     📖 Loaded {wiki_count} Wikipedia articles ✓")

    # 3️⃣ PDF Books
    print("\n 📖 3. Loading PDF books...")
    pdf_docs = load_pdfs(BOOKS_DIR)
    pdf_count = len(pdf_docs)
    raw_docs.extend(pdf_docs)

    # 4️⃣ Chunking
    print("\n 🔪 4. Splitting documents into chunks...")
    chunks = splitter.split_documents(raw_docs)
    print(f"     ✅ Total chunks created: {len(chunks):,}")

    # 5️⃣ FAISS Vector Store
    print("\n 🧮 5. Building FAISS semantic index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VS_DIR)
    print(f"     💾 FAISS saved: {VS_DIR}/index.faiss")

    # 6️⃣ BM25 Keyword Index
    print(" 🔍 6. Building BM25 keyword index...")
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        'bm25': bm25,
        'texts': texts,
        'metas': metas,
        'index_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    bm25_path = os.path.join(VS_DIR, 'bm25.pkl')
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    print(f"     💾 BM25 saved: {bm25_path}")

    # 7️⃣ Save Metadata
    meta = {
        'build_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_chunks': len(chunks),
        'arxiv_count': arxiv_count,
        'wiki_count': wiki_count,
        'pdf_count': pdf_count,
        'embed_model': EMBED_MODEL,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'total_sources': arxiv_count + wiki_count + pdf_count,
        'has_hybrid_search': True
    }
    
    meta_path = os.path.join(VS_DIR, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 8️⃣ Final Report
    print("\n" + "="*70)
    print(" ✅ RAG KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    print("="*70)
    print(f" 📊 TOTAL CHUNKS:        {len(chunks):>4}")
    print(f" 📚 arXiv PAPERS:        {arxiv_count:>4}")
    print(f" 🌐 WIKIPEDIA:          {wiki_count:>4}")
    print(f" 📖 PDF PAGES:          {pdf_count:>4}")
    print(f" 🧮 FAISS INDEX:         ✓")
    print(f" 🔍 BM25 HYBRID:         ✓")
    print(f" 💾 LOCATION:           {VS_DIR}")
    print("="*70)
    print("\n 🚀 Next: `python rag_agents.py`")
    print("   💡 Test: Add PDFs to `rag/books/` and rebuild!")

if __name__ == "__main__":
    build()

# =========================
# Imports (ONLY ONCE)
# =========================
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import torch
import re

# =========================
# MUST BE FIRST STREAMLIT CALL
# =========================
st.set_page_config(
    page_title="YouTube Video Summarizer",
    layout="wide"
)

def clean_transcript(text, words_per_line=20, lines_per_paragraph=4):
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    lines = []
    paragraphs = []

    for i in range(0, len(words), words_per_line):
        line = " ".join(words[i:i + words_per_line])
        lines.append(line.capitalize())

        if len(lines) == lines_per_paragraph:
            paragraphs.append(" ".join(lines))
            lines = []

    if lines:
        paragraphs.append(" ".join(lines))

    return "\n\n".join(paragraphs)

# =========================
# Helper: Extract Video ID
# =========================
def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    Raises ValueError if video ID cannot be found.
    """
    parsed = urlparse(url)

    # Case 1: Standard YouTube URL
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        video_ids = qs.get("v")
        if video_ids:
            return video_ids[0]

    # Case 2: Shortened youtu.be URL
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/")
        if video_id:
            return video_id

    raise ValueError(f"No video id found in URL: {url}")

# =========================
# Get YouTube Transcript
# =========================
@st.cache_data
def get_youtube_transcript(video_url: str, languages=("en", "ar")) -> str:
    """
    Extract and clean the transcript text from a YouTube video URL.
    """
    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=list(languages))
        text = "\n".join(snippet.text for snippet in fetched)

        # Optional cleaning step
        clean_text = clean_transcript(text)

        return clean_text

    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

# =========================
# Load Summarization Model
# =========================
@st.cache_resource
@st.cache_resource
def load_summarizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    
    # Load tokenizer with use_fast=False to avoid SentencePiece conversion issues
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        framework="pt"
    )

# =========================
# Summarize Long Text
# =========================
def summarize_long_text(text: str, summarizer):
    """
    Summarize long text using mT5 by chunking and prefixing.
    """
    chunk_size = 512  # SAFE for mT5
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []

    for chunk in chunks:
        result = summarizer(
            "summarize: " + chunk,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        summaries.append(result[0]["generated_text"])

    return " ".join(summaries)


# =========================
# Main App
# =========================
def main():
    st.title("🎬 Smart YouTube Video Summarizer")

    st.markdown(
        "**Enter a YouTube video URL to extract the transcript and summarize its content using a Transformer model.**"
    )

    st.divider()

    youtube_url = st.text_input("🔗 Enter YouTube URL")

    if st.button("✨ Summarize Video", type="primary"):
        if not youtube_url:
            st.warning("⚠️ Please enter a YouTube URL")
            return

        try:
            summarizer = load_summarizer()

            with st.spinner("⏳ Extracting transcript..."):
                transcript_text = get_youtube_transcript(youtube_url)

            if len(transcript_text) < 50:
                st.warning("Transcript too short or unavailable.")
                return

            with st.spinner("🧠 Summarizing..."):
                final_summary = summarize_long_text(transcript_text, summarizer)

            st.success("✅ Summarization successful!")
            st.subheader("📝 Final Summary")
            st.info(final_summary)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================
# Run App
# =========================
if __name__ == "__main__":
    main()



    🎬 Smart YouTube Video Summarizer
     Overview : 
      A web-based application that automatically extracts transcripts from YouTube videos and generates concise summaries using **Transformer-based NLP models**.
      The app supports **multilingual videos (English & Arabic)** and is built with **Streamlit** for an interactive user experience.

    🚀 Features
          * 🔗 Accepts standard and shortened YouTube URLs
          * 📝 Automatically fetches video transcripts
          * 🌍 Supports **English & Arabic subtitles**
          * 🧹 Cleans and formats transcripts for readability
          * 🧠 Summarizes long videos using **mT5 Transformer model**
          * ⚡ Handles long transcripts via smart chunking
          * 🖥️ Simple and interactive web interface

    🧠 Model Used
         * **Model:** `csebuetnlp/mT5_multilingual_XLSum`
         * **Architecture:** mT5 (Multilingual Text-to-Text Transformer)
         * **Task:** Text Summarization
         * **Why mT5?**
              * Multilingual support
              * Strong performance on abstractive summarization
              * Suitable for long-form content when chunked properly

     🛠️ Tech Stack
          * **Python**
          * **Streamlit** – Web UI
          * **Hugging Face Transformers** – NLP pipeline
          * **PyTorch** – Model backend
          * **YouTube Transcript API** – Subtitle extraction


      📂 Project Structure
           ├── app.py               # Main Streamlit application
           ├── requirements.txt     # Project dependencies
           ├── README.md            # Project documentation


      📦 Requirements
          1. streamlit
          2. youtube-transcript-api
          3. transformers
          4. torch
          5. huggingface-hub
          6. requests


    🧪 How It Works
          1. User enters a **YouTube video URL**
          2. App extracts the **video ID**
          3. Transcript is fetched (English / Arabic)
          4. Transcript is **cleaned & formatted**
          5. Long text is **split into chunks**
          6. Each chunk is summarized using **mT5**
          7. Chunk summaries are combined into a **final summary**


     ⚠️ Notes & Limitations
          * First run may take longer due to **model download**
          * Performance depends on transcript availability
          * Very long videos may take additional processing time
          * CPU-only inference is used for stability



    🌱 Future Improvements
         * Add language selection in UI
         * Enable GPU support (optional)
         * Improve chunking logic (sentence-based)
         * Add summary length control
         * Deploy publicly (Streamlit Cloud / Hugging Face Spaces)

     👤 Author
        **Gasser Ahmed**
        Machine Learning & NLP Enthusiast

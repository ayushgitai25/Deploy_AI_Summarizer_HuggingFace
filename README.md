---
title: "AI Document Summarizer"
emoji: "🔍"
colorFrom: "blue"
colorTo: "green"
sdk: "streamlit"
sdk_version: "1.26.0"
app_file: "app.py"
pinned: false
---

# 🔍 AI Document Summarizer - Streamlit Application

## WHAT THIS APP DOES TO GET AI SUMMARY

### 1. INPUT PROCESSING
- Accepts 3 input types: PDF documents, Website URLs, YouTube videos
- Extracts raw text content from each source type
- Validates and preprocesses the input data

### 2. TEXT EXTRACTION & LOADING
- **PDF**: Uses PyPDFLoader to extract text from uploaded PDF files  
- **Website**: Uses UnstructuredURLLoader to scrape and clean web content  
- **YouTube**: Uses YouTubeTranscriptApi to fetch video transcripts. Fallback: 
  ```python
  from langchain_community.document_loaders import YoutubeLoader
  loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)

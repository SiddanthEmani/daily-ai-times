# GROQ LLM API Rate Limits Reference

This document outlines the rate limits for various Language Model APIs and services offered by GROQ.

## Chat Completion Models

### High-Performance Models

**Large Parameter Models**
- **deepseek-r1-distill-llama-70b**: 30 requests/minute, 1,000 requests/day, 6,000 tokens/minute, unlimited daily tokens
- **llama-3.3-70b-versatile**: 30 requests/minute, 1,000 requests/day, 12,000 tokens/minute, 100,000 tokens/day
- **llama3-70b-8192**: 30 requests/minute, 14,400 requests/day, 6,000 tokens/minute, 500,000 tokens/day

### Standard Models

**Llama Family**
- **llama-3.1-8b-instant**: 30 requests/minute, 14,400 requests/day, 6,000 tokens/minute, 500,000 tokens/day
- **llama3-8b-8192**: 30 requests/minute, 14,400 requests/day, 6,000 tokens/minute, 500,000 tokens/day

**Gemma Models**
- **gemma2-9b-it**: 30 requests/minute, 14,400 requests/day, 15,000 tokens/minute, 500,000 tokens/day

**Specialized Models**
- **allam-2-7b**: 30 requests/minute, 7,000 requests/day, 6,000 tokens/minute, unlimited daily tokens
- **mistral-saba-24b**: 30 requests/minute, 1,000 requests/day, 6,000 tokens/minute, 500,000 tokens/day

### Experimental Models

**Llama 4 Series**
- **meta-llama/llama-4-maverick-17b-128e-instruct**: 30 requests/minute, 1,000 requests/day, 6,000 tokens/minute, unlimited daily tokens
- **meta-llama/llama-4-scout-17b-16e-instruct**: 30 requests/minute, 1,000 requests/day, 30,000 tokens/minute, unlimited daily tokens

**Qwen Models**
- **qwen-qwq-32b**: 30 requests/minute, 1,000 requests/day, 6,000 tokens/minute, unlimited daily tokens
- **qwen/qwen3-32b**: 60 requests/minute, 1,000 requests/day, 6,000 tokens/minute, unlimited daily tokens

### Safety and Guard Models

**Meta Guard Models**
- **llama-guard-3-8b**: 30 requests/minute, 14,400 requests/day, 15,000 tokens/minute, 500,000 tokens/day
- **meta-llama/llama-guard-4-12b**: 30 requests/minute, 14,400 requests/day, 15,000 tokens/minute, 500,000 tokens/day

**Prompt Guard Models**
- **meta-llama/llama-prompt-guard-2-22m**: 30 requests/minute, 14,400 requests/day, 15,000 tokens/minute, unlimited daily tokens
- **meta-llama/llama-prompt-guard-2-86m**: 30 requests/minute, 14,400 requests/day, 15,000 tokens/minute, unlimited daily tokens

## Speech Services

### Speech To Text

All speech-to-text models share the same rate limits:
- **Requests**: 20 per minute, 2,000 per day
- **Audio Processing**: 7,200 seconds per hour, 28,800 seconds per day

**Available Models:**
- **distil-whisper-large-v3-en**: English-optimized Whisper model
- **whisper-large-v3**: Standard Whisper large model
- **whisper-large-v3-turbo**: Faster Whisper variant

### Text To Speech

All text-to-speech models share the same rate limits:
- **Requests**: 10 per minute, 100 per day
- **Tokens**: 1,200 per minute, 3,600 per day

**Available Models:**
- **playai-tts**: Standard text-to-speech
- **playai-tts-arabic**: Arabic language support

## Usage Guidelines

### Planning Your Usage
- Monitor both request and token limits to avoid hitting quotas
- Consider using faster models for development and testing
- Reserve high-capacity models for production workloads

### Rate Limit Considerations
- Models with unlimited daily tokens are better for batch processing
- Higher requests-per-minute limits support real-time applications
- Speech services have lower limits and should be used judiciously

### Model Selection Tips
- Consider token limits when processing large documents

---
Last updated : June 19 2025
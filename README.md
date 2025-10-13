# AI_Learning
This repository showcases my journey to teaching myself AI and ML.

# AI Learning Journey
## Day 1: PyCharm Setup
After experiencing unresolvable issues with VS Code, I switched to PyCharm for Python/Ollama.
-Tokenized Performance Management text from IRM 6.430.1 to CSV.
-Ran local LLM (Llama 3.2) for an explanation of the Four Step Model.
Next: RAG with IRS documents.

## Day 2: RAG Setup
Oh man, talk about nightmares. Lots of errors, but with Grok's help, and a bit of outside-the-box thinking, I was able to 
resolve them all.
-Processed IRM 6.430.1 with NLTK/Pandas (tokenized, saved key terms).
-Built basic RAG with LlamaIndex, queried IRM for Introduction to Performance Management.
Next: Expand RAG with more documents, improve prompt engineering.

## Day 3: Enhanced prompts templates with NLP elements, and a basic agent for chained tasks like quizzes/sims.
I ran into some more errors and had to switch things around a bit from my first draft with Grok.
-Fixed ReActAgent init for version compatability.
-Switched to async .run for ReActAgent compatability with Mistral.
-Added 300-second timeout for Ollama in agent (took almost 20 minutes to run, but it worked).

## Day 4: Expanded RAG with multiple documents, added Streamlit UI for interactive queries.
So this actually took me two days because I kept running into issues with Streamlit not wanting to properly initialize/load Ollama.
-Indexed multiple documents.
-Built Streamlit UI for IRS leadership assistant.
-Fixed asyncio event loop error where Ollama wasn't loading with direct API call.
-Added NLP sentiment analysis.
-Created Q&A pairs from IRMs for fine-tuning.
-Validated JSONL format with NLP analysis.
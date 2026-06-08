# Persona Prompting Effects in LLM-Based Scholar Recommendation
Auditing LLMs as people recommender systems with personas prompting

## LLMs

### Requirements
* Data must be available at `data/context/...`.
* Add API keys for gpt an gemini in `.env`.
* Install packages (see requirements.txt).

### Execution
* Run ollama: `ollama_requests_multiprocessing.py` (see args for details)
* Run gpt/gemini:
  1. `gpt_gemini_01_create_batch_files.py` (modify language using args; uncomment models ony by one)
  2. `gpt_gemini_02_create_batches.py` (modify language and model family using args)
  3. `gpt_gemini_03_retrieve_results.py` (modify model family using args)


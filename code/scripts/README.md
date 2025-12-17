# Batch Processing Scripts

This repository contains two Python scripts for generating and managing prompts in multiple languages: `batch_params.py` and `batch_prompt.py`.

## Overview

These scripts work together to create localized prompt combinations:

1. **`batch_params.py`** - Must be run first. Translates parameter files (instructions, locations, inputs) into the target language.
2. **`batch_prompt.py`** - Run after params. Generates prompt combinations from the translated parameters.
3. **`batch_parse_results.py`** - Run after collecting data from LLMs. Unifies all responses into a `recommendations.csv` and `summary.csv` files.

## Prerequisites
`
Before running either script, set the `PYTHONPATH`. If you are inside `code/scripts` then:
```bash
export PYTHONPATH="$PYTHONPATH:../libs"
```

## batch_params.py

Generates all English params, and translates them to a target language using OpenAI's API.

### Parameters

- `-l, --language` (optionsl): Target language code. Must be one of the supported languages defined in `cons.LANGUAGES`. Default: `cons.LANG_EN` (english).
- `-o, --output-dir` (required): Output directory. Default: `../data/context/`

### Configuration

Requires a `config.ini` file with OpenAI API credentials at `../../config.ini`.

### Example Usage
```bash
# Translate parameters to German
python batch_params.py -l german -o ../data/context/

# Translate parameters to Spanish
python batch_params.py -l spanish -o ../data/context/
```

## batch_prompt.py

Generates prompt combinations from the translated parameter files.

### Parameters

- `-c, --combination_id` (optional): The combination ID to display (0-indexed).
- `-l, --language` (optional): Target language code (must match the language used in `batch_params.py`).
- `-o, --output-dir` (required): Output directory. Default: `../data/context/`

### Example Usage
```bash
# See total combinations available
python batch_prompt.py -c 0 -l german -o ../data/context/

# Display combination ID 5
python batch_prompt.py -c 5 -l german

# Display combination ID 42 in Spanish
python batch_prompt.py -c 42 -l spanish
```


## batch_parse_results.py

- `--results_dir` (required): Directory where all the responses (`.json` files) are located, eg. `../results`
- `--output_dir` (required): Output directory.
- `--model` (optional): Name of the model to load (see list of models under `data/context/models.txt`)
- `--language` (optional): Language to load (`spanish`, `english`, and `german`)


### Example Usage
```bash
# Load and parse all results across sources/models and languages in parallel
nice -n 10 parallel -j 20 python batch_parse_results.py --results_dir ../../results --output_dir ../../results/summary_parallel --model {1} --language {2} :::: ../../data/context/models.txt ::: english german spanish
```


## Complete Workflow Example

Here's a complete example for German language with combination ID 0:
```bash
# Step 1: Set PYTHONPATH
export PYTHONPATH="$PYTHONPATH:../libs"

# Step 2: Translate parameters to German
python batch_params.py -l de -o ../data/context/

# Step 3: Generate and view prompt combination 0
python batch_prompt.py -c 0 -l de -o ../data/context/
```

## Parallel Processing with GNU Parallel

To process multiple combinations in parallel using GNU Parallel:
```bash
# Process combinations 0-719 in parallel (8 jobs at a time)
parallel -j 8 python batch_prompt.py -c {} -l germam -o ../data/context/ ::: {0..719}

# Process all combinations for multiple languages
parallel -j 4 python batch_params.py -l {} -o ../data/context/ ::: english german spanish
parallel -j 8 python batch_prompt.py -c {1} -l {2} ::: {0..719} -o ../data/context/ ::: english german spanish

# Save output to separate files
parallel -j 8 "python batch_prompt.py -c {} -l german -o ../data/context/ > output_{}.txt" ::: {0..719}

# Process with progress bar
parallel --bar -j 8 python batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}
```

## Output

- `batch_params.py` creates translated JSON files in `{output_dir}/{language}/`:
  - `instructions.json`
  - `locations.json`
  - `input.json`

- `batch_prompt.py` creates:
  - `all_prompts.txt` - Contains all generated prompt combinations
  - Console output showing the selected combination's instructions, input, and full prompt

## Notes

- Always run `batch_params.py` before `batch_prompt.py` for each language
- The combination ID in `batch_prompt.py` must be within the valid range (check output for total combinations)
- GNU Parallel is not required but highly recommended for processing multiple combinations efficiently
# LLM Prompting for Emotion Analysis (Q4)

This directory contains the implementation for Q4: Using LLMs to analyze conversations for emotion intensity, empathy, and polarity.

## Overview

The script uses **Llama 3.1 70B** via **Groq API** to analyze 5 conversations with different prompting strategies:

1. **Zero-Shot**: Direct task without examples
2. **Few-Shot**: Provides example analyses before the task
3. **Chain-of-Thought**: Guides step-by-step reasoning

## Setup

### 1. Install Required Package

```bash
pip install groq
```

### 2. Set Your Groq API Key

```bash
export GROQ_API_KEY='your-groq-api-key-here'
```

Or add it to your shell profile (~/.zshrc or ~/.bashrc):

```bash
echo 'export GROQ_API_KEY="your-groq-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Run the Script

```bash
cd Scripts
python llm_prompting.py
```

## Output Files

The script generates two files:

1. **`LLM_output.txt`**: Human-readable text file with all prompts and responses
2. **`LLM_output.json`**: Structured JSON format for programmatic access

## Conversations Analyzed

The script analyzes 5 diverse conversations:

1. **Job Loss** - Friend dealing with unexpected termination
2. **Family Argument** - Conflict resolution with sibling
3. **Promotion Celebration** - Positive news sharing
4. **Exam Anxiety** - Comforting someone under stress
5. **Cancelled Vacation** - Dealing with disappointment

Each conversation has 10-11 utterances (5+ turns).

## Analysis Metrics

For each conversation, the LLM provides:

- **Emotion Intensity** (0-5 scale): How strong the emotions are
- **Empathy Score** (0-5 scale): Level of empathetic response
- **Emotional Polarity**: Positive, Negative, Neutral, or Mixed
- **Reasoning**: Explanation of the analysis

## Prompting Strategies

### Zero-Shot
Direct instruction without examples. Tests the model's inherent understanding.

```
Analyze the following conversation and provide:
1. Emotion Intensity Score (0-5)
2. Empathy Score (0-5)
3. Emotional Polarity (Positive/Negative/Neutral/Mixed)
```

### Few-Shot
Provides 2 example analyses to guide the model.

```
Example 1: [conversation] -> [analysis]
Example 2: [conversation] -> [analysis]
Now analyze: [target conversation]
```

### Chain-of-Thought
Guides the model through step-by-step reasoning.

```
Step 1: Identify main emotions
Step 2: Evaluate intensity (0-5)
Step 3: Assess empathy (0-5)
Step 4: Determine polarity
Step 5: Provide final analysis
```

## Model Details

- **Model**: llama-3.1-70b-versatile
- **Provider**: Groq (fast inference)
- **Temperature**: 0.3 (balanced creativity/consistency)
- **Max Tokens**: 1000

## Expected Runtime

- ~15-30 seconds total (Groq API is very fast)
- 3 API calls per conversation × 5 conversations = 15 total calls

## Example Output Structure

```
CONVERSATION 1
Context: A friend just lost their job
================================================================================

Utterances:
  1. I can't believe they let me go after 5 years.
  2. I'm so sorry to hear that. That must be really tough.
  ...

--------------------------------------------------------------------------------
Zero-Shot Prompting
--------------------------------------------------------------------------------
Response:
{
  "emotion_intensity": 4,
  "empathy": 5,
  "polarity": "Negative",
  "reasoning": "High emotional intensity due to job loss shock..."
}
```

## Troubleshooting

### API Key Not Found
```
❌ ERROR: GROQ_API_KEY not found in environment variables
```
**Solution**: Set the environment variable as shown in Setup step 2.

### Rate Limiting
If you hit rate limits, the script will show an error. Wait a moment and retry.

### Import Error
```
ModuleNotFoundError: No module named 'groq'
```
**Solution**: Run `pip install groq`

## Notes

- The script uses structured JSON responses for easier parsing
- All conversations are predefined and diverse in emotional content
- The output file is suitable for human analysis and comparison
- Temperature is set low (0.3) for more consistent results across runs

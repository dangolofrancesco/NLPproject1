import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def zero_shot_prompt(conversation):
    utterances_text = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(conversation['utterances'])])
    
    prompt = f"""Analyze the following conversation and provide:
1. Emotion Intensity Score (0-5 scale, where 0=no emotion, 5=very intense emotion)
2. Empathy Score (0-5 scale, where 0=no empathy, 5=high empathy)
3. Emotional Polarity (0=Neutral, 1=Positive, 2=Negative, 3=Mixed)

Conversation (ID: {conversation['id']}, Article: {conversation['article_id']}):
{utterances_text}

Provide your analysis in JSON format with keys: emotion_intensity, empathy, polarity, and reasoning."""

    return prompt


def few_shot_prompt(conversation):
    utterances_text = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(conversation['utterances'])])
    
    prompt = f"""Analyze conversations for emotion intensity, empathy, and polarity.

Example 1:
Conversation: 
1. Person 1: I'm so excited! I got accepted to my dream school!
2. Person 2: That's wonderful news! I'm thrilled for you!
Analysis: {{"emotion_intensity": 5, "empathy": 5, "polarity": 1, "reasoning": "Very high positive emotion with strong empathetic response"}}

Example 2:
Conversation: 
1. Person 1: I'm worried about the test.
2. Person 2: You'll do fine, don't stress.
Analysis: {{"emotion_intensity": 3, "empathy": 3, "polarity": 3, "reasoning": "Moderate anxiety with supportive but brief response"}}

Now analyze this conversation:

Conversation (ID: {conversation['id']}, Article: {conversation['article_id']}):
{utterances_text}

Provide your analysis in JSON format with keys: emotion_intensity, empathy, polarity (0=Neutral, 1=Positive, 2=Negative, 3=Mixed), and reasoning."""

    return prompt


def chain_of_thought_prompt(conversation):
    utterances_text = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(conversation['utterances'])])
    
    prompt = f"""Analyze the following conversation step by step:

Conversation (ID: {conversation['id']}, Article: {conversation['article_id']}):
{utterances_text}

Please think through this systematically:

Step 1: Identify the main emotions expressed in the conversation.
Step 2: Evaluate the intensity of these emotions on a 0-5 scale.
Step 3: Assess how much empathy is shown in responses (0-5 scale).
Step 4: Determine the overall emotional polarity (0=Neutral, 1=Positive, 2=Negative, 3=Mixed).
Step 5: Provide your final analysis.

Format your response as JSON with keys: emotion_intensity, empathy, polarity, reasoning, and thinking_process."""

    return prompt


def analyze_with_llm(prompt, model="llama-3.3-70b-versatile"):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=1000,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

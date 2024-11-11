
import requests

def get_llm_response(prompt: str) -> str | None:
    """Returns the LLM's response to `prompt`."""
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer sk-gwLqBEGDDIQUwFph2sKwNkqyUkTp-O74C0Yifh9zupT3BlbkFJUjZW5wq4bfgwnKng6aEIZ0xRuc915JB-Aw5uzHgNwA"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 2000
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        message = data['choices'][0]['message']['content']
        return message.strip()
    else:
        print(f"Error: Received status code {response.status_code}")
        return None


def get_disease_text(text: str) -> str:
    """Returns the LLM's interpretation of the next `n` prime numbers, 
    in ascending order, that are no smaller than `lower_bound`, 
    as a comma separated list."""

    answer = get_llm_response(f"What is the most likely disease for {text}")
    if answer is None:
        print(f"Unknown error. LLM returned None.")
        return None
    return answer

get_disease_text("hello I have a rash")
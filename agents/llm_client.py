
import os
import json


def call_llm(system_prompt: str, user_prompt: str, fallback: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return fallback
    try:
        import urllib.request
        import urllib.error

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) trust-geoagent/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return fallback + f"\n\n[LLM call failed, HTTP {e.code}: {body}]"
    except Exception as e:  # noqa: BLE001 - demo-grade fallback is intentional
        return fallback + f"\n\n[LLM call failed, using fallback: {e}]"
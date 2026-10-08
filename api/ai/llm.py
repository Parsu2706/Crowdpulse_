from functools import lru_cache
from api.config import GROQ_API_KEY , GROQ_MODEL

class LLMUnavailable(RuntimeError):
    pass

@lru_cache(maxsize=1)
def client():
    if not GROQ_API_KEY:
        raise LLMUnavailable("GROQ_API_KEY is not set")

    from groq import Groq
    return Groq(api_key =GROQ_API_KEY)

def chat(prompt : str , *, system : str | None = None , json_mode : bool = False , 
         max_tokens : int = 800 , temp : float = 0.2) -> str:
    messages = ([{"role" : "system" , "content" : system}] if system else []) + \
                [{"role" : "user" , "content" : prompt}]
    kwargs = {"response_format" : {"type" : "json_object"}} if json_mode else {}
    resp = client().chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=max_tokens,
        temperature=temp, **kwargs,
    )
    return resp.choices[0].message.content.strip()

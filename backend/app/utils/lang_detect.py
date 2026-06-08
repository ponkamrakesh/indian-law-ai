from langdetect import detect, LangDetectException
from typing import Dict

# Supported Indian languages mapping (langdetect codes)
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    # Add more as needed
}

def detect_language(text: str) -> Dict[str, str]:
    """Detect language of user query. Returns code and name."""
    try:
        if len(text.strip()) < 10:
            return {"code": "en", "name": "English"}
        lang_code = detect(text)
        lang_name = LANGUAGE_MAP.get(lang_code, "English")
        return {"code": lang_code, "name": lang_name}
    except LangDetectException:
        return {"code": "en", "name": "English"}
    except Exception:
        return {"code": "en", "name": "English"}
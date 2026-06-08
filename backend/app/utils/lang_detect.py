from langdetect import detect, LangDetectException
from typing import Dict

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
}

def detect_language(text: str) -> Dict[str, str]:
    try:
        if len(text.strip()) < 10:
            return {"code": "en", "name": "English"}
        lang_code = detect(text)
        lang_name = LANGUAGE_MAP.get(lang_code, "English")
        return {"code": lang_code, "name": lang_name}
    except:
        return {"code": "en", "name": "English"}
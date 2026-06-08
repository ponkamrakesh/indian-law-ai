from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool
import re

@tool
def get_youtube_transcript(url: str) -> str:
    """Get transcript from a YouTube video URL for legal research."""
    try:
        video_id = re.search(r"(?:v=|youtu.be/)([^&]+)", url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t['text'] for t in transcript])
        return text[:3000]  # Limit length for efficiency
    except Exception as e:
        return f"Could not fetch transcript: {str(e)}"
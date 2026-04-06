from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

url: str = os.getenv("SUPABASE_URL", "")
key: str = os.getenv("SUPABASE_ANON_KEY", "")

try:
    if url and key and url.startswith("http"):
        supabase: Client = create_client(url, key)
    else:
        supabase = None
except Exception:
    supabase = None

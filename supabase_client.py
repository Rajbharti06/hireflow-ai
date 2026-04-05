from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

url: str = os.getenv("SUPABASE_URL", "")
key: str = os.getenv("SUPABASE_ANON_KEY", "")

# Initialize client only if keys are present
if url and key:
    supabase: Client = create_client(url, key)
else:
    # We will let it fail gracefully or mock if missing keys during initial setup
    supabase = None

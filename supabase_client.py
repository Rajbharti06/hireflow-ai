from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL", "")
anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

try:
    if url and anon_key and url.startswith("http"):
        # Auth client — uses anon key, session is patched per-user on every run
        supabase: Client = create_client(url, anon_key)
        # DB client — uses service role key (bypasses RLS entirely).
        # If no service key is set, falls back to the anon client (RLS applies).
        supabase_db: Client = (
            create_client(url, service_key)
            if service_key
            else supabase
        )
    else:
        supabase = None
        supabase_db = None
except Exception:
    supabase = None
    supabase_db = None

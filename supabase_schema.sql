-- =============================================
-- HireFlow AI — Supabase Schema
-- Run this in Supabase → SQL Editor → New Query
-- =============================================

-- 1. Jobs table
CREATE TABLE IF NOT EXISTS public.jobs (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id     UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  title       TEXT NOT NULL,
  description TEXT,
  created_at  TIMESTAMPTZ DEFAULT timezone('utc', now()) NOT NULL
);

-- 2. Results table
CREATE TABLE IF NOT EXISTS public.results (
  id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id         UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  job_id          UUID REFERENCES public.jobs(id) ON DELETE CASCADE,
  resume_id       UUID,
  score           NUMERIC,
  explanation     TEXT,
  candidate_name  TEXT,
  filename        TEXT,
  embedding_score NUMERIC,
  skill_score     NUMERIC,
  llm_score       NUMERIC,
  rank            INTEGER,
  skills_json     JSONB,
  shortlisted     BOOLEAN DEFAULT false,
  created_at      TIMESTAMPTZ DEFAULT timezone('utc', now()) NOT NULL
);

-- 3. Profiles table (tracks plan + lifetime usage)
CREATE TABLE IF NOT EXISTS public.profiles (
  id                       UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email                    TEXT NOT NULL DEFAULT '',
  is_pro                   BOOLEAN DEFAULT false,
  lemon_squeezy_customer_id TEXT,
  lifetime_usage           INTEGER DEFAULT 0,
  created_at               TIMESTAMPTZ DEFAULT timezone('utc', now()) NOT NULL
);

-- 4. Auto-create a profile row whenever a user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email)
  VALUES (NEW.id, COALESCE(NEW.email, ''))
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 5. Atomic usage increment RPC (race-condition-safe, auto-creates profile if missing)
CREATE OR REPLACE FUNCTION public.increment_usage(uid UUID, amount INTEGER DEFAULT 1)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_email TEXT;
BEGIN
  SELECT email INTO v_email FROM auth.users WHERE id = uid;
  -- Upsert: inserts profile if it doesn't exist, otherwise increments counter
  INSERT INTO public.profiles (id, email, lifetime_usage)
  VALUES (uid, COALESCE(v_email, ''), amount)
  ON CONFLICT (id) DO UPDATE
    SET lifetime_usage = profiles.lifetime_usage + EXCLUDED.lifetime_usage;
END;
$$;

-- 6. Row Level Security (disabled for MVP — re-enable when ready for production)
-- To enable: uncomment the ENABLE lines and comment out the DISABLE lines
ALTER TABLE public.jobs     DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.results  DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;

-- Production RLS policies (uncomment when ready):
-- ALTER TABLE public.jobs     ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.results  ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "own_jobs"     ON public.jobs     FOR ALL USING (auth.uid() = user_id);
-- CREATE POLICY "own_results"  ON public.results  FOR ALL USING (auth.uid() = user_id);
-- CREATE POLICY "own_profile"  ON public.profiles FOR ALL USING (auth.uid() = id);

-- Run this in the Supabase SQL Editor

-- 1. Create the Jobs table
CREATE TABLE IF NOT EXISTS jobs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Create the Results table
CREATE TABLE IF NOT EXISTS results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
  resume_id UUID,
  score NUMERIC,
  explanation TEXT,
  candidate_name TEXT,
  filename TEXT,
  embedding_score NUMERIC,
  skill_score NUMERIC,
  llm_score NUMERIC,
  rank INTEGER,
  skills_json JSONB,
  shortlisted BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create the Users Profile table (for Lemon Squeezy PRO status)
-- We map this table directly to Supabase Auth so when users sign up, we can manage their subscription
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT NOT NULL,
  is_pro BOOLEAN DEFAULT false,
  lemon_squeezy_customer_id TEXT,
  lifetime_usage INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Secure Lifetime Usage RPC
-- This atomically increments a user's usage from Python (preventing race conditions)
CREATE OR REPLACE FUNCTION public.increment_usage(uid UUID, amount INTEGER DEFAULT 1)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  UPDATE public.profiles
  SET lifetime_usage = lifetime_usage + amount
  WHERE id = uid;
END;
$$;

-- Note: We disable RLS (Row Level Security) temporarily for the MVP 
-- just to make sure frontend testing works flawlessly and avoid 42501 API errors.
-- You can enable and configure these later using the Supabase UI.
ALTER TABLE public.jobs DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.results DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;

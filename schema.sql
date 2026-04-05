-- USERS handled by Supabase Auth

create table jobs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id),
    title text,
    description text,
    created_at timestamp default now()
);

create table resumes (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id),
    job_id uuid references jobs(id),
    name text,
    content text,
    created_at timestamp default now()
);

create table results (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id),
    job_id uuid references jobs(id),
    resume_id uuid references resumes(id),
    score float,
    explanation text,
    
    -- Added fields for full data preservation
    candidate_name text,
    filename text,
    embedding_score float,
    skill_score float,
    llm_score float,
    rank int,
    shortlisted boolean default false,
    skills_json jsonb,
    
    created_at timestamp default now()
);

-- Row Level Security (RLS)
alter table jobs enable row level security;
alter table resumes enable row level security;
alter table results enable row level security;

create policy "Users can access their own jobs"
on jobs for all
using (auth.uid() = user_id);

create policy "Users can access their own resumes"
on resumes for all
using (auth.uid() = user_id);

create policy "Users can access their own results"
on results for all
using (auth.uid() = user_id);

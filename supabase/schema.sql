-- Krish & Associates: website picks. Run once in Supabase → SQL Editor → New query → Run.
-- Safe to re-run. Access rules (RLS) do all the protecting; the site only holds the public key.

create table if not exists public.profiles (
  id uuid primary key references auth.users on delete cascade,
  name text not null check (char_length(name) between 1 and 30),
  created_at timestamptz not null default now()
);

create table if not exists public.kickoffs (
  game_id text primary key,          -- nflverse id, e.g. 2026_08_CAR_GB
  kickoff timestamptz not null
);

create table if not exists public.picks (
  user_id uuid not null default auth.uid() references auth.users on delete cascade,
  game_id text not null references public.kickoffs (game_id),
  p_home real not null check (p_home between 0.01 and 0.99),   -- your chance the home team wins
  updated_at timestamptz not null default now(),
  primary key (user_id, game_id)
);

-- the server stamps every write; a client can't backdate a pick
create or replace function public.stamp_pick() returns trigger language plpgsql as $$
begin new.updated_at := now(); return new; end $$;
drop trigger if exists stamp_pick on public.picks;
create trigger stamp_pick before insert or update on public.picks for each row execute function public.stamp_pick();

create or replace function public.not_started(gid text) returns boolean language sql stable as $$
  select exists (select 1 from public.kickoffs k where k.game_id = gid and k.kickoff > now())
$$;

alter table public.profiles enable row level security;
alter table public.kickoffs enable row level security;
alter table public.picks enable row level security;

drop policy if exists "names are public" on public.profiles;
drop policy if exists "own profile insert" on public.profiles;
drop policy if exists "own profile update" on public.profiles;
create policy "names are public" on public.profiles for select using (true);
create policy "own profile insert" on public.profiles for insert to authenticated with check (auth.uid() = id);
create policy "own profile update" on public.profiles for update to authenticated using (auth.uid() = id) with check (auth.uid() = id);

-- kickoffs are read-only from the site (loaded by kickoffs.sql in the SQL editor)
drop policy if exists "kickoffs are public" on public.kickoffs;
create policy "kickoffs are public" on public.kickoffs for select using (true);

-- your own picks any time; everyone's picks only once that game has kicked off
drop policy if exists "read picks" on public.picks;
drop policy if exists "file before kickoff" on public.picks;
drop policy if exists "change before kickoff" on public.picks;
drop policy if exists "withdraw before kickoff" on public.picks;
create policy "read picks" on public.picks for select
  using (auth.uid() = user_id or not public.not_started(game_id));
create policy "file before kickoff" on public.picks for insert to authenticated
  with check (auth.uid() = user_id and public.not_started(game_id));
create policy "change before kickoff" on public.picks for update to authenticated
  using (auth.uid() = user_id and public.not_started(game_id))
  with check (auth.uid() = user_id and public.not_started(game_id));
create policy "withdraw before kickoff" on public.picks for delete to authenticated
  using (auth.uid() = user_id and public.not_started(game_id));

"""Write supabase/kickoffs.sql: every scheduled game's kickoff for the current season, as an upsert.

    python src/supabase_kickoffs.py
Paste the file into Supabase's SQL Editor after schema.sql, and again whenever the NFL flexes a game
or the playoff schedule is set (the pick lock follows these times).
"""
from common import CURRENT_SEASON, ROOT, kickoff_utc, load_games

OUT = ROOT / "supabase" / "kickoffs.sql"


def main():
    g = load_games()
    g = g[g["season"] == CURRENT_SEASON]
    rows = [f"('{r.game_id}', '{kickoff_utc(r.gameday, r.gametime).isoformat()}')" for r in g.itertuples()]
    OUT.write_text("insert into public.kickoffs (game_id, kickoff) values\n  " + ",\n  ".join(rows)
                   + "\non conflict (game_id) do update set kickoff = excluded.kickoff;\n")
    print(f"{len(rows)} kickoffs -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

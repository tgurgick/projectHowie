#!/usr/bin/env python3
import os, argparse, pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DB_URL", "sqlite:///fantasy.db")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--pos-buckets", default="QB:12,RB:36,WR:48,TE:12,DEF:12,K:12")
    ap.add_argument("--db-url", default=DB_URL)
    args = ap.parse_args()

    buckets = dict([kv.split(":") for kv in args.pos_buckets.split(",")])
    buckets = {k:int(v) for k,v in buckets.items()}

    eng = create_engine(args.db_url, future=True)
    
    # Check if required data exists using direct SQLite
    import sqlite3
    db_path = args.db_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    
    pgs_count = pd.read_sql("SELECT COUNT(*) as count FROM player_game_stats", conn).iloc[0]['count']
    games_count = pd.read_sql("SELECT COUNT(*) as count FROM games", conn).iloc[0]['count']
    
    if pgs_count == 0:
        print(f"❌ No player game stats data found. Run the main ETL script first:")
        print(f"   python build_fantasy_db.py --start {args.season} --end {args.season} --scoring ppr")
        return 1
        
    if games_count == 0:
        print(f"❌ No games data found. Run the main ETL script first:")
        print(f"   python build_fantasy_db.py --start {args.season} --end {args.season} --scoring ppr")
        return 1
    
    # Execute the main query
    query = f"""
        SELECT g.season, p.position, s.player_id, SUM(s.fantasy_points) AS pts
        FROM player_game_stats s
        JOIN games g ON g.game_id=s.game_id
        JOIN players p ON p.player_id=s.player_id
        WHERE g.season={args.season}
        GROUP BY g.season, p.position, s.player_id
    """
    pg = pd.read_sql(query, conn)
    conn.close()

    repl = []
    for pos, k in buckets.items():
        pos_df = pg[pg["position"]==pos].sort_values("pts", ascending=False).reset_index(drop=True)
        if len(pos_df)==0: continue
        idx = min(k-1, len(pos_df)-1)
        repl_pts = float(pos_df.loc[idx, "pts"])
        pos_df["vor"] = pos_df["pts"] - repl_pts
        repl.append(pos_df.assign(replacement_points=repl_pts))
    out = pd.concat(repl, ignore_index=True)
    out.to_sql("vor_season", eng, if_exists="replace", index=False)
    print(f"Wrote VOR table for {args.season} with positions {list(buckets.keys())}")

if __name__ == "__main__":
    main()
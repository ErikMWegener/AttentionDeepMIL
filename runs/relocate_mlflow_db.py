#!/usr/bin/env python
"""Point the artifact paths in runs/mlflow.db to the current repository location.

MLflow stores absolute artifact paths. The results database was written under
/home/erik/AttentionDeepMIL; after copying runs/mlflow.db and runs/mlruns/ into
another checkout, MLflow (UI and client) only finds the artifacts after this
rewrite. A backup runs/mlflow.db.bak is written first. Run from the repository root:

    python runs/relocate_mlflow_db.py [--old /home/erik/AttentionDeepMIL]
"""
import argparse
import os
import shutil
import sqlite3

COLUMNS = [('runs', 'artifact_uri'),
           ('experiments', 'artifact_location'),
           ('logged_models', 'artifact_location')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', default='runs/mlflow.db')
    parser.add_argument('--old', default='/home/erik/AttentionDeepMIL', help='repository path stored in the DB')
    parser.add_argument('--new', default=os.getcwd(), help='new repository path (default: current directory)')
    args = parser.parse_args()

    shutil.copy2(args.db, args.db + '.bak')
    con = sqlite3.connect(args.db)
    for table, col in COLUMNS:
        cur = con.execute(f"UPDATE {table} SET {col} = replace({col}, ?, ?)", (args.old, args.new))
        print(f"{table}.{col}: {cur.rowcount} rows")
    con.commit()
    con.close()

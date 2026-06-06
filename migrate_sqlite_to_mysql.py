"""
migrate_sqlite_to_mysql.py
Run this ONCE to copy your existing SQLite access_events data into MySQL.
Make sure MySQL is configured and running before running this script.

Usage:
    python migrate_sqlite_to_mysql.py
"""
import sqlite3
import os
import db_mysql

SQLITE_PATH = 'logs/access_events.db'


def migrate():
    if not os.path.exists(SQLITE_PATH):
        print(f'[MIGRATE] No SQLite file found at {SQLITE_PATH} — nothing to migrate.')
        return

    print('[MIGRATE] Initialising MySQL schema...')
    db_mysql.init_schema()

    print(f'[MIGRATE] Reading from {SQLITE_PATH}...')
    old = sqlite3.connect(SQLITE_PATH)
    old.row_factory = sqlite3.Row

    # Check which columns exist in the old DB
    cols_info = old.execute('PRAGMA table_info(access_events)').fetchall()
    col_names = [c[1] for c in cols_info]
    print(f'[MIGRATE] Old columns: {col_names}')

    rows = old.execute('SELECT * FROM access_events ORDER BY id').fetchall()
    print(f'[MIGRATE] Found {len(rows)} rows to migrate.')

    conn = db_mysql.get_connection()
    cur  = conn.cursor()

    success = 0
    errors  = 0

    for r in rows:
        try:
            person_name = r['person_name'] if 'person_name' in col_names else 'UNKNOWN'
            person_id   = r['person_id']   if 'person_id'   in col_names else None
            status      = r['status']      if 'status'      in col_names else 'AUTHORIZED'
            entry_time  = r['entry_time']  if 'entry_time'  in col_names else None
            exit_time   = r['exit_time']   if 'exit_time'   in col_names else None
            duration_s  = r['duration_s']  if 'duration_s'  in col_names else None
            snapshot    = r['snapshot']    if 'snapshot'    in col_names else None
            camera      = r['camera']      if 'camera'      in col_names else 'webcam'

            cur.execute("""
                INSERT INTO access_events
                    (person_name, person_id, status, entry_time, exit_time,
                     duration_s, snapshot, camera_label, location)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (person_name, person_id, status, entry_time, exit_time,
                  duration_s, snapshot, camera, None))
            success += 1
        except Exception as e:
            print(f'[MIGRATE] Row error: {e}')
            errors += 1

    conn.commit()
    cur.close()
    conn.close()
    old.close()

    print(f'\n[MIGRATE] Done.')
    print(f'  ✅ Migrated : {success} rows')
    if errors:
        print(f'  ⚠  Errors  : {errors} rows (see above)')
    print('\nYou can now run the application normally with MySQL.')


if __name__ == '__main__':
    migrate()

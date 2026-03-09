from contextlib import contextmanager

import config

import psycopg

@contextmanager
def db_cursor(host, user, password, dbname=config.db_name, port=5432, autocommit=False):
    conn = psycopg.connect(host=host, user=user, password=password, dbname=dbname, port=port)
    cursor = conn.cursor()
    try:
        yield cursor
        if autocommit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        # Always rollback. If autocommit=True, then the
        # transaction will have already been commited, so this is "failsafe"
        conn.rollback()
        cursor.close()
        conn.close()

def preevents_cursor(readonly=True, autocommit=False):
    """
    Simple wrapper for the db_cursor context manager, defaulting all values
    with a simple flag to switch between read-only and read-write user.
    """
    user = config.db_read_user if readonly else config.db_write_user
    password = config.db_read_pass if readonly else config.db_write_pass
    return db_cursor(config.db_host, user, password, autocommit=autocommit)


def interpret_rections(signals: set[str]):
    # Early exits / special cases
    if not signals:
        return None, None    
    if '-1' in signals:
        return False, None
    if 'question' in signals:
        return None, 'ambiguous'
    if signals == {'+1'}:
        return True, 'volcanic'

    # Lookup tables
    source_map = {
        'volcano': 'volcanic',
        'tea': 'lake',
        'fire': 'other',
    }

    source = source_map[(signals - {'+1'}).pop()]
    true_positive = True if signals else None

    return true_positive, source
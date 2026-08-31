from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import config

import pandas
import psycopg
import redis

from hotlink import support_functions

REDIS_DB = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


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

@lru_cache(maxsize=None)
def load_volcs():
    # Load volcanoes from the PREEVENTS database
    with preevents_cursor() as cursor:
        cursor.execute("""
                       SELECT longitude    as lon,
                              latitude     as lat,
                              volcano_name as name,
                              elevation    as elev,
                              volcano_id   as id
                       FROM volcano
                       WHERE observatory = 'avo'
                       """)

        columns = [desc.name for desc in cursor.description]
        data = pandas.DataFrame(cursor.fetchall(), columns=columns)

    return data

@dataclass(frozen=True)
class Volcano:
    id: int
    name: str
    lat: float
    lon: float
    elev: int

    @property
    def coords(self) -> tuple[float, float]:
        return self.lat, self.lon


@lru_cache(None)
def get_volc(vent: str | list | tuple) -> Volcano:
    VOLCS = load_volcs()
    if isinstance(vent, str):
        volc = VOLCS[VOLCS['name'].str.lower() == vent.lower()]
        if len(volc) == 0:
            raise ValueError(f"Specified volcano ({vent}) not found!. Candidates:\n{sorted(VOLCS['name'])}")
    else:
        dists = support_functions.haversine_np(vent[1], vent[0], VOLCS['lon'], VOLCS['lat'])
        volc = VOLCS[dists == dists.min()]

    row = volc.iloc[0]
    return Volcano(
        id=row.id,
        name=str(row["name"]),
        lat=row.lat,
        lon=row.lon,
        elev=row.elev,
    )
import logging
import pickle
import time
from pathlib import Path
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

_volc_cache = None
_last_check = 0

def load_volcs():
    """
    Load volcanoes from the PREEVENTS database, with a persistent disk-based cache.
    Checks for updates once a day. If it can't get new data, it uses the existing cache.
    """
    global _volc_cache, _last_check

    cache_file = Path(__file__).parent / 'data' / 'volcano_cache.pkl'
    cache_expiration = 86400  # 1 day in seconds
    now = time.time()

    # 1. In-memory cache check (within the same process)
    if _volc_cache is not None and (now - _last_check) < cache_expiration:
        return _volc_cache

    # 2. Disk cache check
    mtime = cache_file.stat().st_mtime if cache_file.exists() else 0
    if (now - mtime) < cache_expiration:
        try:
            with cache_file.open('rb') as f:
                _volc_cache = pickle.load(f)
            _last_check = mtime  # Set to actual data age
            return _volc_cache
        except Exception as e:
            logging.warning(f"Failed to load volcano cache from {cache_file}: {e}")
            # Fall through to DB refresh

    # 3. Database refresh
    try:
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

        # Save to disk cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open('wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.warning(f"Failed to save volcano cache to {cache_file}: {e}")

        _volc_cache = data
        _last_check = now
        return data

    except Exception as db_err:
        logging.error(f"Failed to load volcanoes from database: {db_err}")

        # 4. Fallback: use in-memory cache if we have it from earlier in this process
        if _volc_cache is not None:
            logging.info("Using in-memory volcano cache as fallback after DB failure.")
            return _volc_cache

        # 5. Fallback: use the existing disk cache (even if expired)
        if mtime > 0:
            try:
                with cache_file.open('rb') as f:
                    data = pickle.load(f)
                _volc_cache = data
                # Note: We do NOT update _last_check here so that we keep trying the DB
                # on subsequent requests (per user request).
                logging.info(f"Using existing disk volcano cache as fallback after DB failure.")
                return data
            except Exception as e:
                logging.error(f"Failed to load volcano cache fallback: {e}")

        # If all failed and no cache available, raise
        raise db_err

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
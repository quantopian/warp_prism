from contextlib import contextmanager
from uuid import uuid4
import warnings

import psycopg2


def _dropdb(cur, db_name):
    cur.execute('COMMIT')
    cur.execute('DROP DATABASE %s' % db_name)


@contextmanager
def tmp_db_uri():
    """Create a temporary postgres database to run the tests against.
    """
    db_name = '_warp_prism_test_' + uuid4().hex
    root = 'postgresql://localhost/'
    uri = root + db_name
    with psycopg2.connect(root + 'postgres') as conn, conn.cursor() as cur:
        cur.execute('COMMIT')
        cur.execute('CREATE DATABASE %s' % db_name)
        try:
            yield uri
        finally:
            try:
                cur.execute("""
                    select
                        pg_terminate_backend(pid)
                    from
                        pg_stat_activity
                    where
                        pid != pg_backend_pid()
                """)
                _dropdb(cur, db_name)
            except:  # pragma: no cover  # noqa
                # The database wasn't cleaned up. Just tell the user to deal
                # with this manually.
                warnings.warn(
                    "leaking database '%s', please manually delete this" %
                    db_name,
                )

from contextlib import contextmanager
from uuid import uuid4
import warnings

from odo import resource
import sqlalchemy as sa


def _dropdb(root_conn, db_name):
    root_conn.execute('COMMIT')
    root_conn.execute('DROP DATABASE %s' % db_name)


@contextmanager
def disposable_engine(uri):
    """An engine which is disposed on exit.

    Parameters
    ----------
    uri : str
        The uri to the db.

    Yields
    ------
    engine : sa.engine.Engine
    """
    engine = resource(uri)
    try:
        yield engine
    finally:
        engine.dispose()


_pg_stat_activity = sa.Table(
    'pg_stat_activity',
    sa.MetaData(),
    sa.Column('pid', sa.Integer),
)


@contextmanager
def tmp_db_uri():
    """Create a temporary postgres database to run the tests against.
    """
    db_name = '_warp_prism_test_' + uuid4().hex
    root = 'postgresql://localhost/'
    uri = root + db_name
    with disposable_engine(root + 'postgres') as e, e.connect() as root_conn:
        root_conn.execute('COMMIT')
        root_conn.execute('CREATE DATABASE %s' % db_name)
        try:
            yield uri
        finally:
            resource(uri).dispose()
            try:
                _dropdb(root_conn, db_name)
            except sa.exc.OperationalError:
                # We couldn't drop the db. The most likely cause is that there
                # are active queries. Even more likely is that these are
                # rollbacks because there was an exception somewhere inside the
                # tests. We will cancel all the running queries and try to drop
                # the database again.
                pid = _pg_stat_activity.c.pid
                root_conn.execute(
                    sa.select(
                        (sa.func.pg_terminate_backend(pid),),
                    ).where(
                        pid != sa.func.pg_backend_pid(),
                    )
                )
                try:
                    _dropdb(root_conn, db_name)
                except sa.exc.OperationalError:  # pragma: no cover
                    # The database STILL wasn't cleaned up. Just tell the user
                    # to deal with this manually.
                    warnings.warn(
                        "leaking database '%s', please manually delete this" %
                        db_name,
                    )

from contextlib import contextmanager
from string import ascii_letters
from uuid import uuid4
import warnings

from datashape import var, R, Option, dshape
import numpy as np
from odo import resource, odo
import pandas as pd
import pytest
import sqlalchemy as sa

from warp_prism import (
    to_arrays,
    to_dataframe,
    null_values as null_values_for_type,
)


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


@pytest.fixture(scope='module')
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


@pytest.fixture
def tmp_table_uri(tmp_db_uri):
    return '%s::%s%s' % (tmp_db_uri, 'table_', uuid4().hex)


def check_roundtrip_nonnull(table_uri, data, dtype, sqltype):
    """Check the data roundtrip through postgres using warp_prism to read the
    data

    Parameters
    ----------
    table_uri : str
        The uri to a unique table.
    data : np.array
        The input data.
    dtype : str
        The dtype of the data.
    sqltype : type
        The sqlalchemy type of the data.
    """
    input_dataframe = pd.DataFrame({'a': data})
    table = odo(input_dataframe, table_uri, dshape=var * R['a': dtype])
    assert table.columns.keys() == ['a']
    assert isinstance(table.columns['a'].type, sqltype)

    arrays = to_arrays(table)
    assert len(arrays) == 1
    array, mask = arrays['a']
    assert (array == data).all()
    assert mask.all()

    output_dataframe = to_dataframe(table)
    assert (output_dataframe == input_dataframe).all().all()


@pytest.mark.parametrize('dtype,sqltype,start,stop,step', (
    ('int16', sa.SmallInteger, 0, 5000, 1),
    ('int32', sa.Integer, 0, 5000, 1),
    ('int64', sa.BigInteger, 0, 5000, 1),
    ('float32', sa.REAL, 0, 2500, 0.5),
    ('float64', sa.FLOAT, 0, 2500, 0.5),
))
def test_numeric_type_nonnull(tmp_table_uri,
                              dtype,
                              sqltype,
                              start,
                              stop,
                              step):
    data = np.arange(start, stop, step, dtype=dtype)
    check_roundtrip_nonnull(tmp_table_uri, data, dtype, sqltype)


def test_bool_type_nonnull(tmp_table_uri):
    data = np.array([True] * 2500 + [False] * 2500, dtype=bool)
    check_roundtrip_nonnull(tmp_table_uri, data, 'bool', sa.Boolean)


def test_string_type_nonnull(tmp_table_uri):
    data = np.array(list(ascii_letters) * 200, dtype='object')
    check_roundtrip_nonnull(tmp_table_uri, data, 'string', sa.String)


def test_datetime_type_nonnull(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        np.timedelta64(1, 'D'),
        dtype='datetime64[us]',
    )
    check_roundtrip_nonnull(tmp_table_uri, data, 'datetime', sa.DateTime)


def test_date_type_nonnull(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        dtype='datetime64[D]',
    )
    check_roundtrip_nonnull(tmp_table_uri, data, 'date', sa.Date)


def check_roundtrip_null_values(table_uri,
                                data,
                                dtype,
                                sqltype,
                                null_values,
                                mask,
                                *,
                                astype=False):
    """Check the data roundtrip through postgres using warp_prism to read the
    data

    Parameters
    ----------
    table_uri : str
        The uri to a unique table.
    data : iterable[any]
        The input data.
    dtype : str
        The dtype of the data.
    sqltype : type
        The sqlalchemy type of the data.
    null_values : dict[str, any]
        The value to coerce ``NULL`` to.
    astype : bool, optional
        Coerce the input data to the given dtype before making assertions about
        the output data.
    """
    table = resource(table_uri, dshape=var * R['a': Option(dtype)])
    assert table.columns.keys() == ['a']
    assert isinstance(table.columns['a'].type, sqltype)
    table.insert().values([{'a': v} for v in data]).execute()

    arrays = to_arrays(table)
    assert len(arrays) == 1
    array, actual_mask = arrays['a']
    assert (actual_mask == mask).all()
    assert (array[mask] == data[mask]).all()

    output_dataframe = to_dataframe(table, null_values=null_values)
    if astype:
        data = data.astype(dshape(dtype).measure.to_numpy_dtype())
    expected_dataframe = pd.DataFrame({'a': data})
    expected_dataframe[~mask] = null_values.get(
        'a',
        null_values_for_type[
            array.dtype
            if array.dtype.kind != 'M' else
            np.dtype('datetime64[ns]')
        ],
    )
    pd.util.testing.assert_frame_equal(
        output_dataframe,
        expected_dataframe,
        check_dtype=False,
    )


def check_roundtrip_null(table_uri,
                         data,
                         dtype,
                         sqltype,
                         null,
                         mask,
                         *,
                         astype=False):
    """Check the data roundtrip through postgres using warp_prism to read the
    data

    Parameters
    ----------
    table_uri : str
        The uri to a unique table.
    data : iterable[any]
        The input data.
    dtype : str
        The dtype of the data.
    sqltype : type
        The sqlalchemy type of the data.
    null : any
        The value to coerce ``NULL`` to.
    astype : bool, optional
        Coerce the input data to the given dtype before making assertions about
        the output data.
    """
    check_roundtrip_null_values(
        table_uri,
        data,
        dtype,
        sqltype,
        {'a': null},
        mask,
        astype=astype,
    )


@pytest.mark.parametrize('dtype,sqltype,start,stop,step,null', (
    ('int16', sa.SmallInteger, 0, 5000, 1, -1),
    ('int32', sa.Integer, 0, 5000, 1, -1),
    ('int64', sa.BigInteger, 0, 5000, 1, -1),
    ('float32', sa.REAL, 0, 2500, 0.5, -1.0),
    ('float64', sa.FLOAT, 0, 2500, 0.5, -1.0),
))
def test_numeric_type_null(tmp_table_uri,
                           dtype,
                           sqltype,
                           start,
                           stop,
                           step,
                           null):
    data = np.arange(start, stop, step, dtype=dtype).astype(object)
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(tmp_table_uri, data, dtype, sqltype, null, mask)


@pytest.mark.parametrize('dtype,sqltype', (
    ('int16', sa.SmallInteger),
    ('int32', sa.Integer),
    ('int64', sa.BigInteger),
))
def test_numeric_default_null_promote(tmp_table_uri, dtype, sqltype):
    data = np.arange(0, 100, dtype=dtype).astype(object)
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null_values(tmp_table_uri, data, dtype, sqltype, {}, mask)


def test_bool_type_null(tmp_table_uri):
    data = np.array([True] * 2500 + [False] * 2500, dtype=bool).astype(object)
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(tmp_table_uri, data, 'bool', sa.Boolean, False, mask)


def test_string_type_null(tmp_table_uri):
    data = np.array(list(ascii_letters) * 200, dtype='object')
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(
        tmp_table_uri,
        data,
        'string',
        sa.String,
        'ayy lmao',
        mask,
    )


def test_datetime_type_null(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        np.timedelta64(1, 'D'),
        dtype='datetime64[us]',
    ).astype(object)
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(
        tmp_table_uri,
        data,
        'datetime',
        sa.DateTime,
        np.datetime64('1995-12-13', 'ns'),
        mask,
    )


def test_date_type_null(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        dtype='datetime64[D]',
    ).astype(object)
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(
        tmp_table_uri,
        data,
        'date',
        sa.Date,
        np.datetime64('1995-12-13', 'ns'),
        mask,
        astype=True,
    )

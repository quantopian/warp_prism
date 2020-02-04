from string import ascii_letters
import struct
from uuid import uuid4

import numpy as np
import psycopg2
import pytest

from warp_prism._warp_prism import (
    postgres_signature,
    raw_to_arrays,
    test_overflow_operations as _test_overflow_operations,
)
from warp_prism import to_arrays, to_dataframe
from warp_prism.query import null_values as null_values_for_type
from warp_prism.types import dtype_to_typeid
from warp_prism.tests import tmp_db_uri as tmp_db_uri_ctx

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(scope='module')
def tmp_db_uri():
    with tmp_db_uri_ctx() as db_uri:
        yield db_uri


@pytest.fixture
def tmp_table_uri(tmp_db_uri):
    return '%s::%s%s' % (tmp_db_uri, 'table_', uuid4().hex)


def item(a):
    """Convert a value to a Python built in type (not a numpy type).

    Parameters
    ----------
    a : any
        The value to convert.

    Returns
    -------
    item : any
        The base Python type equivalent value.
    """
    try:
        return a.item()
    except AttributeError:
        return a


def check_roundtrip_nonnull(table_uri, data, dtype, sqltype):
    """Check the data roundtrip through postgres using warp_prism to read the
    data

    Parameters
    ----------
    table_uri : str
        The uri for the  table.
    data : np.array
        The input data.
    dtype : str
        The dtype of the data.
    sqltype : str
        The sql type of the data.
    """
    db, table = table_uri.split('::')
    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute('create table %s (a %s)' % (table, sqltype))
        cur.executemany(
            'insert into {} values (%s)'.format(table),
            [(item(v),) for v in data],
        )

        query = 'select * from %s' % table
        arrays = to_arrays(query, bind=conn)

        if pd is not None:
            output_dataframe = to_dataframe(query, bind=conn)

    assert len(arrays) == 1
    array, mask = arrays['a']
    assert (array == data).all()
    assert mask.all()

    if pd is not None:
        expected_dataframe = pd.DataFrame({'a': data})
        pd.util.testing.assert_frame_equal(
            output_dataframe,
            expected_dataframe,
        )


@pytest.mark.parametrize('dtype,sqltype,start,stop,step', (
    ('int16', 'int2', 0, 5000, 1),
    ('int32', 'int4', 0, 5000, 1),
    ('int64', 'int8', 0, 5000, 1),
    ('float32', 'float4', 0, 2500, 0.5),
    ('float64', 'float8', 0, 2500, 0.5),
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
    check_roundtrip_nonnull(tmp_table_uri, data, 'bool', 'bool')


def test_string_type_nonnull(tmp_table_uri):
    data = np.array(list(ascii_letters) * 200, dtype='object')
    check_roundtrip_nonnull(tmp_table_uri, data, 'object', 'text')


def test_datetime_type_nonnull(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        dtype='M8[D]',
    ).astype('datetime64[us]')
    check_roundtrip_nonnull(tmp_table_uri, data, 'datetime64[us]', 'timestamp')


def test_date_type_nonnull(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        dtype='M8[D]',
    ).values.astype('datetime64[D]')
    check_roundtrip_nonnull(tmp_table_uri, data, 'datetime64[D]', 'date')


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
    sqltype : str
        The sql type of the data.
    null_values : dict[str, any]
        The value to coerce ``NULL`` to.
    astype : bool, optional
        Coerce the input data to the given dtype before making assertions about
        the output data.
    """
    db, table = table_uri.split('::')
    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute('create table %s (a %s)' % (table, sqltype))
        cur.executemany(
            'insert into {} values (%s)'.format(table),
            [(item(v),) for v in data],
        )

        query = 'select * from %s' % table
        arrays = to_arrays(query, bind=conn)
        if pd is not None:
            output_dataframe = to_dataframe(
                query,
                null_values=null_values,
                bind=conn,
            )

    assert len(arrays) == 1
    array, actual_mask = arrays['a']
    assert (actual_mask == mask).all()

    assert (array[mask] == data[mask]).all()

    if pd is not None:
        if astype:
            data = data.astype(dtype, copy=False)

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
    sqltype : str
        The sql type of the data.
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
    ('int16', 'int2', 0, 5000, 1, -1),
    ('int32', 'int4', 0, 5000, 1, -1),
    ('int64', 'int8', 0, 5000, 1, -1),
    ('float32', 'float4', 0, 2500, 0.5, -1.0),
    ('float64', 'float8', 0, 2500, 0.5, -1.0),
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
    ('int16', 'int2'),
    ('int32', 'int4'),
    ('int64', 'int8'),
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
    check_roundtrip_null(tmp_table_uri, data, 'bool', 'bool', False, mask)


def test_string_type_null(tmp_table_uri):
    data = np.array(list(ascii_letters) * 200, dtype='object')
    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(
        tmp_table_uri,
        data,
        'object',
        'text',
        'ayy lmao',
        mask,
    )


def test_datetime_type_null(tmp_table_uri):
    data = np.arange(
        '2000',
        '2016',
        dtype='M8[D]',
    ).astype('O')[:-1]  # slice the last element off to have an even number

    mask = np.tile(np.array([True, False]), len(data) // 2)
    data[~mask] = None
    check_roundtrip_null(
        tmp_table_uri,
        data,
        'datetime64[us]',
        'timestamp',
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
        'datetime64[D]',
        'date',
        np.datetime64('1995-12-13', 'ns'),
        mask,
        astype=True,
    )


def _pack_as_invalid_size_postgres_binary_data(char, itemsize, value):
    """Create mock postgres data for testing the column data size checks.

    Parameters
    ----------
    char : str
        The format char for struct.
    value : any
        The value to pack, this will appear twice.

    Returns
    -------
    binary_data : bytes
        The binary data to feed to raw_to_arrays.
    """
    return postgres_signature + struct.pack(
        '>iihi{char}hi{char}'.format(char=char),
        0,  # flags
        0,  # extension area size
        1,  # field_count
        itemsize,  # data_size
        value,
        1,  # field_count
        itemsize - 1,  # incorrect size for the given type
        value,  # default value of the given type
    )


@pytest.mark.parametrize('dtype', map(np.dtype, (
    'bool',
    'int16',
    'int32',
    'float32',
    'float64',
)))
def test_invalid_numeric_size(dtype):
    input_data = _pack_as_invalid_size_postgres_binary_data(
        dtype.char,
        dtype.itemsize,
        dtype.type(),
    )

    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, (dtype_to_typeid(dtype),))

    assert str(e.value) == 'mismatched %s size: %s' % (
        dtype.name,
        dtype.itemsize - 1,
    )


# timedelta to adjust a numpy datetime into a postgres datetime
_epoch_offset = np.datetime64('2000-01-01') - np.datetime64('1970-01-01')


def test_invalid_datetime_size():
    input_data = _pack_as_invalid_size_postgres_binary_data(
        'q',  # int64_t (quadword)
        8,
        (np.datetime64('2014-01-01', 'us') + _epoch_offset).view('int64'),
    )

    dtype = np.dtype('datetime64[us]')
    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, (dtype_to_typeid(dtype),))

    assert str(e.value) == 'mismatched datetime size: 7'


def test_invalid_date_size():
    input_data = _pack_as_invalid_size_postgres_binary_data(
        'i',  # int32_t
        4,
        (np.datetime64('2014-01-01', 'D') + _epoch_offset).view('int64'),
    )

    dtype = np.dtype('datetime64[D]')
    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, (dtype_to_typeid(dtype),))

    assert str(e.value) == 'mismatched date size: 3'


def test_invalid_text():
    input_data = postgres_signature + struct.pack(
        '>iihi1si1shi{}si1s'.format(len(postgres_signature)),
        0,  # flags
        0,  # extension area size

        # row 0
        2,  # field_count
        1,  # data_size
        b'\0',
        1,  # data_size
        b'\0',

        # row 1
        2,  # field_count
        len(postgres_signature) + 1,  # data_size
        postgres_signature + b'\0',  # postgres signature is invalid unicode
        1,  # data_size
        b'\1',
    )
    # we put the invalid unicode as the first column to test that we can clean
    # up the cell in the second column before we have written a string there

    str_typeid = dtype_to_typeid(np.dtype(object))
    with pytest.raises(UnicodeDecodeError):
        raw_to_arrays(input_data, (str_typeid, str_typeid))


def test_missing_signature():
    input_data = b''

    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, ())

    assert str(e.value) == 'missing postgres signature'


def test_missing_flags():
    input_data = postgres_signature

    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, ())

    assert (
        str(e.value) == 'reading 4 bytes would cause an out of bounds access'
    )


def test_missing_extension_length():
    input_data = postgres_signature + (0).to_bytes(4, 'big')

    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, ())

    assert (
        str(e.value) == 'reading 4 bytes would cause an out of bounds access'
    )


def test_missing_end_marker():
    input_data = postgres_signature + (0).to_bytes(4, 'big')

    with pytest.raises(ValueError) as e:
        raw_to_arrays(input_data, ())

    assert (
        str(e.value) == 'reading 4 bytes would cause an out of bounds access'
    )


def test_overflow_operations():
    # thanks pytest
    _test_overflow_operations()

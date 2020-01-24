import numpy as np

from ._warp_prism import typeid_map as _raw_typeid_map

_typeid_map = {np.dtype(k): v for k, v in _raw_typeid_map.items()}


def dtype_to_typeid(dtype):
    """Convert a numpy dtype to a warp_prism type id.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype to convert.

    Returns
    -------
    typeid : int
        The type id for ``dtype``.
    """
    try:
        return _typeid_map[dtype]
    except KeyError:
        raise ValueError('no warp_prism type id for dtype %s' % dtype)


_oid_map = {
    16: np.dtype('?'),

    # text
    17: np.dtype('O'),
    18: np.dtype('S1'),
    19: np.dtype('O'),
    25: np.dtype('O'),

    # int
    20: np.dtype('i8'),
    21: np.dtype('i2'),
    23: np.dtype('i4'),
    1042: np.dtype('O'),
    1043: np.dtype('O'),

    # float
    700: np.dtype('f4'),
    701: np.dtype('f8'),

    # date(time)
    1082: np.dtype('M8[D]'),
    1114: np.dtype('M8[us]'),
    1184: np.dtype('M8[us]'),
}


def oid_to_dtype(oid):
    """Get a numpy dtype from postgres oid.

    Parameters
    ----------
    oid : int
        The oid to convert.

    Returns
    -------
    dtype : np.dtype
        The corresponding numpy dtype.
    """
    try:
        return _oid_map[oid]
    except KeyError:
        raise ValueError('cannot convert oid %s to numpy dtype' % oid)


def query_dtypes(cursor, bound_query):
    """Get the numpy dtypes for each column returned by a query.

    Parameters
    ----------
    cursor : psycopg2.cursor
       The psycopg2 cursor to use to get the type information.
    bound_query : str
       The query to check the types of with all parameters bound.

    Returns
    -------
    names : tuple[str]
        The column names.
    dtypes : tuple[np.dtype]
        The column dtypes.
    """
    cursor.execute('select * from (%s) a limit 0' % bound_query)
    invalid = []
    names = []
    dtypes = []
    for c in cursor.description:
        try:
            dtypes.append(oid_to_dtype(c.type_code))
        except ValueError:
            invalid.append(c)
        else:
            names.append(c.name)

    if invalid:
        raise ValueError(
            'columns cannot be converted to numpy dtype: %s' % invalid
        )

    return tuple(names), tuple(dtypes)


def query_typeids(cursor, bound_query):
    """Get the warp_prism typeid for each column returned by a query.

    Parameters
    ----------
    cursor : psycopg2.cursor
       The psycopg2 cursor to use to get the type information.
    bound_query : str
       The query to check the types of with all parameters bound.

    Returns
    -------
    names : tuple[str]
        The column names.
    typeids : tuple[int]
        The warp_prism typeid for each column..
    """
    names, dtypes = query_dtypes(cursor, bound_query)
    return names, tuple(dtype_to_typeid(dtype) for dtype in dtypes)

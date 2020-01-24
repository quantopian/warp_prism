from functools import wraps
import io

try:
    import pandas as pd
except ImportError:
    pd = None
import numpy as np

from .sql import getbind, mogrify
from .types import query_typeids
from ._warp_prism import raw_to_arrays as _raw_to_arrays


def to_arrays(query, params=None, *, bind=None):
    """Run the query returning a the results as np.ndarrays.

    Parameters
    ----------
    query : str or sa.sql.Selectable
        The query to run. This can be a select or a table.
    params : dict or tuple or None
        Bind parameters for ``query``.
    bind : psycopg2.connection, sa.Engine, or sa.Connection, optional
        The engine used to create the connection. If not provided
        ``query.bind`` will be used.

    Returns
    -------
    arrays : dict[str, (np.ndarray, np.ndarray)]
        A map from column name to the result arrays. The first array holds the
        values and the second array is a boolean mask for NULLs. The values
        where the mask is False are 0 interpreted by the type.
    """

    buf = io.BytesIO()
    bind = getbind(query, bind)

    with bind.cursor() as cur:
        bound_query = mogrify(cur, query, params)
        column_names, typeids = query_typeids(cur, bound_query)
        cur.copy_expert('copy (%s) to stdout binary' % bound_query, buf)

    out = _raw_to_arrays(buf.getbuffer(), typeids)

    return {column_names[n]: v for n, v in enumerate(out)}


null_values = {np.dtype(k): v for k, v in {
    'float32': np.nan,
    'float64': np.nan,
    'int16': np.nan,
    'int32': np.nan,
    'int64': np.nan,
    'bool': np.nan,
    'datetime64[ns]': np.datetime64('nat', 'ns'),
    'object': None,
}.items()}

# alias because ``to_dataframe`` shadows this name
_default_null_values_for_type = null_values


def to_dataframe(query, params=None, *, bind=None, null_values=None):
    """Run the query returning a the results as a pd.DataFrame.

    Parameters
    ----------
    query : str or sa.sql.Selectable
        The query to run. This can be a select or a table.
    params : dict or tuple or None
        Bind parameters for ``query``.
    bind : psycopg2.connection, sa.Engine, or sa.Connection, optional
        The engine used to create the connection. If not provided
        ``query.bind`` will be used.
    null_values : dict[str, any]
        The null values to use for each column. This falls back to
        ``warp_prism.null_values`` for columns that are not specified.

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame holding the results of the query. The columns
        of the DataFrame will be named the same and be in the same order as the
        query.
    """
    arrays = to_arrays(query, bind=bind)

    if null_values is None:
        null_values = {}

    for name, (array, mask) in arrays.items():
        if array.dtype.kind == 'i':
            if not mask.all():
                try:
                    null = null_values[name]
                except KeyError:
                    # no explicit override, cast to float and use NaN as null
                    array = array.astype('float64')
                    null = np.nan

                array[~mask] = null

            arrays[name] = array
            continue

        if array.dtype.kind == 'M':
            # pandas needs datetime64[ns], not ``us`` or ``D``
            array = array.astype('datetime64[ns]')

        try:
            null = null_values[name]
        except KeyError:
            null = _default_null_values_for_type[array.dtype]

        array[~mask] = null
        arrays[name] = array

    return pd.DataFrame(arrays)


if pd is None:
    @wraps(to_dataframe)
    def to_dataframe(*args, **kwargs):
        raise NotImplementedError('to_dataframe requires pandas')

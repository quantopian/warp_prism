from io import BytesIO
import numbers

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as _postgresql
from sqlalchemy.ext.compiler import compiles

from ._warp_prism import (
    raw_to_arrays as _raw_to_arrays,
    typeid_map as _raw_typeid_map,
)


__version__ = '0.1.1'


_typeid_map = {np.dtype(k): v for k, v in _raw_typeid_map.items()}
_object_type_id = _raw_typeid_map['object']


class _CopyToBinary(sa.sql.expression.Executable, sa.sql.ClauseElement):

    def __init__(self, element, bind):
        self.element = element
        self._bind = bind = bind

    @property
    def bind(self):
        return self._bind


def literal_compile(s):
    """Compile a sql expression with bind params inlined as literals.

    Parameters
    ----------
    s : Selectable
        The expression to compile.

    Returns
    -------
    cs : str
        An equivalent sql string.
    """
    return str(s.compile(compile_kwargs={'literal_binds': True}))


@compiles(_CopyToBinary, 'postgresql')
def _compile_copy_to_binary_postgres(element, compiler, **kwargs):
    selectable = element.element
    return compiler.process(
        sa.text(
            'COPY {stmt} TO STDOUT (FORMAT BINARY)'.format(
                stmt=(
                    compiler.preparer.format_table(selectable)
                    if isinstance(selectable, sa.Table) else
                    '({})'.format(literal_compile(selectable))
                ),
            )
        ),
        **kwargs
    )


types = {np.dtype(k): v for k, v in {
    'i8': sa.BigInteger,
    'i4': sa.Integer,
    'i2': sa.SmallInteger,
    'f4': sa.REAL,
    'f8': sa.FLOAT,
    'O': sa.Text,
    'M8[D]': sa.Date,
    'M8[us]': sa.DateTime,
    '?': sa.Boolean,
    "m8[D]": sa.Interval(second_precision=0, day_precision=9),
    "m8[h]": sa.Interval(second_precision=0, day_precision=0),
    "m8[m]": sa.Interval(second_precision=0, day_precision=0),
    "m8[s]": sa.Interval(second_precision=0, day_precision=0),
    "m8[ms]": sa.Interval(second_precision=3, day_precision=0),
    "m8[us]": sa.Interval(second_precision=6, day_precision=0),
    "m8[ns]": sa.Interval(second_precision=9, day_precision=0),
}.items()}

_revtypes = dict(map(reversed, types.items()))
_revtypes.update({
    sa.DATETIME: np.dtype('M8[us]'),
    sa.TIMESTAMP: np.dtype('M8[us]'),
    sa.FLOAT: np.dtype('f8'),
    sa.DATE: np.dtype('M8[D]'),
    sa.BIGINT: np.dtype('i8'),
    sa.INTEGER: np.dtype('i4'),
    sa.BIGINT: np.dtype('i8'),
    sa.types.NullType: np.dtype('O'),
    sa.REAL: np.dtype('f4'),
    sa.Float: np.dtype('f8'),
})

_precision_types = {
    sa.Float,
    _postgresql.base.DOUBLE_PRECISION,
}


def _precision_to_dtype(precision):
    if isinstance(precision, numbers.Integral):
        if 1 <= precision <= 24:
            return np.dtype('f4')
        elif 25 <= precision <= 53:
            return np.dtype('f8')
    raise ValueError('%s is not a supported precision' % precision)


_units_of_power = {
    0: 's',
    3: 'ms',
    6: 'us',
    9: 'ns'
}


def _discover_type(type_):
    if isinstance(type_, sa.Interval):
        if type_.second_precision is None and type_.day_precision is None:
            return np.dtype('m8[us]')
        elif type_.second_precision == 0 and type_.day_precision == 0:
            return np.dtype('m8[s]')

        if (type_.second_precision in _units_of_power and
                not type_.day_precision):
            unit = _units_of_power[type_.second_precision]
        elif type_.day_precision > 0:
            unit = 'D'
        else:
            raise ValueError(
                'Cannot infer INTERVAL type_e with parameters'
                'second_precision=%d, day_precision=%d' %
                (type_.second_precision, type_.day_precision),
            )
        return np.dtype('m8[%s]' % unit)
    if type(type_) in _precision_types and type_.precision is not None:
        return _precision_to_dtype(type_.precision)
    if type_ in _revtypes:
        return _revtypes[type_]
    if type(type_) in _revtypes:
        return _revtypes[type(type_)]
    if isinstance(type_, sa.Numeric):
        raise ValueError('Cannot adapt numeric type to numpy dtype')
    if isinstance(type_, (sa.String, sa.Unicode)):
        return np.dtype('O')
    else:
        for k, v in _revtypes.items():
            if isinstance(k, type) and (isinstance(type_, k) or
                                        hasattr(type_, 'impl') and
                                        isinstance(type_.impl, k)):
                return v
            if k == type_:
                return v
    raise NotImplementedError('No SQL-numpy match for type %s' % type_)


def _warp_prism_types(query):
    for col in query.columns:
        dtype = _discover_type(col.type)
        try:
            yield _typeid_map[dtype]
        except KeyError:
            raise TypeError(
                'warp_prism cannot query columns of type %s' % dtype,
            )


def _getbind(selectable, bind):
    """Return an explicitly passed connection or infer the connection from
    the selectable.

    Parameters
    ----------
    selectable : sa.sql.Selectable
        The selectable object being queried.
    bind : bind or None
        The explicit connection or engine to use to execute the query.

    Returns
    -------
    bind : The bind which should be used to execute the query.
    """
    if bind is None:
        return selectable.bind

    if isinstance(bind, sa.engine.base.Engine):
        return bind

    return sa.create_engine(bind)


def to_arrays(query, *, bind=None):
    """Run the query returning a the results as np.ndarrays.

    Parameters
    ----------
    query : sa.sql.Selectable
        The query to run. This can be a select or a table.
    bind : sa.Engine, optional
        The engine used to create the connection. If not provided
        ``query.bind`` will be used.

    Returns
    -------
    arrays : dict[str, (np.ndarray, np.ndarray)]
        A map from column name to the result arrays. The first array holds the
        values and the second array is a boolean mask for NULLs. The values
        where the mask is False are 0 interpreted by the type.
    """
    # check types before doing any work
    types = tuple(_warp_prism_types(query))

    buf = BytesIO()
    bind = _getbind(query, bind)

    stmt = _CopyToBinary(query, bind)
    with bind.connect() as conn:
        conn.connection.cursor().copy_expert(literal_compile(stmt), buf)
    out = _raw_to_arrays(buf.getbuffer(), types)
    column_names = query.c.keys()
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


def to_dataframe(query, *, bind=None, null_values=None):
    """Run the query returning a the results as a pd.DataFrame.

    Parameters
    ----------
    query : sa.sql.Selectable
        The query to run. This can be a select or a table.
    bind : sa.Engine, optional
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

    return pd.DataFrame(arrays, columns=[column.name for column in query.c])


def register_odo_dataframe_edge():
    """Register an odo edge for sqlalchemy selectable objects to dataframe.

    This edge will have a lower cost that the default edge so it will be
    selected as the fasted path.

    If the selectable is not in a postgres database, it will fallback to the
    default odo edge.
    """
    from odo import convert
    from datashape.predicates import istabular

    # estimating 8 times faster
    df_cost = convert.graph.edge[sa.sql.Select][pd.DataFrame]['cost'] / 8

    @convert.register(
        pd.DataFrame,
        (sa.sql.Select, sa.sql.Selectable),
        cost=df_cost,
    )
    def select_or_selectable_to_frame(el, bind=None, dshape=None, **kwargs):
        bind = _getbind(el, bind)

        if bind.dialect.name != 'postgresql':
            # fall back to the general edge
            raise NotImplementedError()

        return to_dataframe(el, bind=bind)

    # higher priority than df edge so that
    # ``odo('select one_column from ...', list)``  returns a list of scalars
    # instead of a list of tuples of length 1
    @convert.register(
        pd.Series,
        (sa.sql.Select, sa.sql.Selectable),
        cost=df_cost - 1,
    )
    def select_or_selectable_to_series(el, bind=None, dshape=None, **kwargs):
        bind = _getbind(el, bind)

        if istabular(dshape) or bind.dialect.name != 'postgresql':
            # fall back to the general edge
            raise NotImplementedError()

        return to_dataframe(el, bind=bind).iloc[:, 0]

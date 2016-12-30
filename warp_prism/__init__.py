from io import BytesIO

from datashape import discover
from datashape.predicates import istabular
import numpy as np
from odo.backends.sql import getbind
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from toolz import keymap

from ._warp_prism import to_arrays as _raw_to_arrays, typeid_map as _typeid_map


__version__ = '0.1.0'

typeid_map = keymap(np.dtype, _typeid_map)
null_values = keymap(np.dtype, {
    'float32': np.nan,
    'float64': np.nan,
    'int32': np.nan,
    'int64': np.nan,
    'datetime64[ns]': np.datetime64('nat', 'ns'),
    'object': None,
})


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


def _warp_prism_types(query):
    for name, dtype in discover(query).measure.fields:
        try:
            yield typeid_map[getattr(dtype, 'ty', dtype).to_numpy_dtype()]
        except KeyError:
            raise TypeError(
                'warp_prism cannot query columns of type %s' % dtype,
            )


def to_arrays(query, bind=None):
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
    bind = getbind(query, bind)

    stmt = _CopyToBinary(query, bind)
    with bind.connect() as conn:
        conn.connection.cursor().copy_expert(literal_compile(stmt), buf)
    out = _raw_to_arrays(buf.getbuffer(), types)
    column_names = query.c.keys()
    return {column_names[n]: v for n, v in enumerate(out)}


def to_dataframe(query, bind=None):
    """Run the query returning a the results as a pd.DataFrame.

    Parameters
    ----------
    query : sa.sql.Selectable
        The query to run. This can be a select or a table.
    bind : sa.Engine, optional
        The engine used to create the connection. If not provided
        ``query.bind`` will be used.

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame holding the results of the query. The columns
        of the DataFrame will be named the same and be in the same order as the
        query.
    """
    arrays = to_arrays(query, bind=bind)
    for name, (array, mask) in arrays.items():
        if array.dtype.kind == 'i':
            if not mask.all():
                array = array.astype('float64')
                array[~mask] = null_values[array.dtype]
            arrays[name] = array
            continue

        if array.dtype.kind == 'M':
            array = array.astype('datetime64[ns]')

        array[~mask] = null_values[array.dtype]
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

    # estimating 8 times faster
    df_cost = convert.graph.edge[sa.sql.Select][pd.DataFrame]['cost'] / 8

    @convert.register(
        pd.DataFrame,
        (sa.sql.Select, sa.sql.Selectable),
        cost=df_cost,
    )
    def select_or_selectable_to_frame(el, bind=None, dshape=None, **kwargs):
        bind = getbind(el, bind)

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
        bind = getbind(el, bind)

        if istabular(dshape) or bind.dialect.name != 'postgresql':
            # fall back to the general edge
            raise NotImplementedError()

        return to_dataframe(el, bind=bind).iloc[:, 0]

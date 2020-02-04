from .query import to_dataframe
from .sql import getbind


def register_odo_dataframe_edge():
    """Register an odo edge for sqlalchemy selectable objects to dataframe.

    This edge will have a lower cost that the default edge so it will be
    selected as the fasted path.

    If the selectable is not in a postgres database, it will fallback to the
    default odo edge.
    """
    from datashape.predicates import istabular
    from odo import convert
    import pandas as pd
    import sqlalchemy as sa

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

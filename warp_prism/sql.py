import psycopg2
try:
    import sqlalchemy as sa
except ImportError:
    sa = None


def _sa_literal_compile(s):
    """Compile a sql expression with variables inlined as literals.

    Parameters
    ----------
    s : sa.sql.Selectable
        The expression to compile.

    Returns
    -------
    cs : str
        An equivalent sql string.
    """
    return str(s.compile(compile_kwargs={'literal_binds': True}))


def mogrify(cursor, query, params):
    if sa is not None:
        if isinstance(query, sa.Table):
            query = _sa_literal_compile(sa.select(query.c))
        elif isinstance(query, sa.sql.Selectable):
            query = _sa_literal_compile(query)

    return cursor.mogrify(query, params).decode('utf-8')


def getbind(query, bind):
    """Get the connection to use for a query.

    Parameters
    ----------
    query : str or sa.sql.Selectable
        The query to run.
    bind : psycopg2.extensions.connection or sa.engine.base.Engine or None
        The explicitly provided bind.

    Returns
    -------
    bind : psycopg2.extensions.connection
        The connection to use for the query.
    """
    if bind is not None:
        if sa is None or isinstance(bind, psycopg2.extensions.connection):
            return bind

        if isinstance(bind, sa.engine.base.Engine):
            return bind.connect().connection.connection

        return sa.create_engine(bind).connect().connection.connection
    elif sa is None or not isinstance(query, sa.sql.Selectable):
        raise TypeError("missing 1 required argument: 'bind'")
    else:
        return query.bind.connect().connection.connection

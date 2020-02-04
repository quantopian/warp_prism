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

warp_prism
==========

Quickly move data from postgres to numpy or pandas.

API
---

``to_arrays(query, *, bind=None)``
``````````````````````````````````

.. code-block::

   Run the query returning a the results as np.ndarrays.

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


``to_dataframe(query, *, bind=None, null_values=None)``
```````````````````````````````````````````````````````

.. code-block::

   Run the query returning a the results as a pd.DataFrame.

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


``register_odo_dataframe_edge()``
`````````````````````````````````

.. code-block::

   Register an odo edge for sqlalchemy selectable objects to dataframe.

   This edge will have a lower cost that the default edge so it will be
   selected as the fasted path.

   If the selectable is not in a postgres database, it will fallback to the
   default odo edge.


Comparisons
-----------

A quick comparison between ``warp_prism``, ``odo``, and ``pd.read_sql_table``.

In this example we will read real data for VIX from quandl stored in a local
postgres database using ``warp_prism``, ``odo``, and ``pd.read_sql_table``.
After that, we will use ``odo`` to create a table with two float columns and
1000000 rows and query it with the tree tools again.

.. code-block:: python

   In [1]: import warp_prism

   In [2]: from odo import odo, resource

   In [3]: import pandas as pd

   In [4]: table = resource(
      ...:     'postgresql://localhost/bz::yahoo_index_vix',
      ...:     schema='quandl',
      ...: )

   In [5]: warp_prism.to_dataframe(table).head()
   Out[5]:
      asof_date      open_       high        low      close  volume  \
   0 2016-01-08  22.959999  27.080000  22.480000  27.010000     0.0
   1 2015-12-04  17.430000  17.650000  14.690000  14.810000     0.0
   2 2015-10-29  14.800000  15.460000  14.330000  14.610000     0.0
   3 2015-12-21  19.639999  20.209999  18.700001  18.700001     0.0
   4 2015-10-26  14.760000  15.430000  14.680000  15.290000     0.0

      adjusted_close                  timestamp
   0       27.010000 2016-01-11 23:14:54.682220
   1       14.810000 2016-01-11 23:14:54.682220
   2       14.610000 2016-01-11 23:14:54.682220
   3       18.700001 2016-01-11 23:14:54.682220
   4       15.290000 2016-01-11 23:14:54.682220

   In [6]: odo(table, pd.DataFrame).head()
   Out[6]:
      asof_date      open_       high        low      close  volume  \
   0 2016-01-08  22.959999  27.080000  22.480000  27.010000     0.0
   1 2015-12-04  17.430000  17.650000  14.690000  14.810000     0.0
   2 2015-10-29  14.800000  15.460000  14.330000  14.610000     0.0
   3 2015-12-21  19.639999  20.209999  18.700001  18.700001     0.0
   4 2015-10-26  14.760000  15.430000  14.680000  15.290000     0.0

      adjusted_close                  timestamp
   0       27.010000 2016-01-11 23:14:54.682220
   1       14.810000 2016-01-11 23:14:54.682220
   2       14.610000 2016-01-11 23:14:54.682220
   3       18.700001 2016-01-11 23:14:54.682220
   4       15.290000 2016-01-11 23:14:54.682220

   In [7]: pd.read_sql_table(table.name, table.bind, table.schema).head()
   Out[7]:
      asof_date      open_       high        low      close  volume  \
   0 2016-01-08  22.959999  27.080000  22.480000  27.010000     0.0
   1 2015-12-04  17.430000  17.650000  14.690000  14.810000     0.0
   2 2015-10-29  14.800000  15.460000  14.330000  14.610000     0.0
   3 2015-12-21  19.639999  20.209999  18.700001  18.700001     0.0
   4 2015-10-26  14.760000  15.430000  14.680000  15.290000     0.0

      adjusted_close                  timestamp
   0       27.010000 2016-01-11 23:14:54.682220
   1       14.810000 2016-01-11 23:14:54.682220
   2       14.610000 2016-01-11 23:14:54.682220
   3       18.700001 2016-01-11 23:14:54.682220
   4       15.290000 2016-01-11 23:14:54.682220

   In [8]: len(warp_prism.to_dataframe(table))
   Out[8]: 6565

   In [9]: %timeit warp_prism.to_dataframe(table)
   100 loops, best of 3: 7.55 ms per loop

   In [10]: %timeit odo(table, pd.DataFrame)
   10 loops, best of 3: 49.9 ms per loop

   In [11]: %timeit pd.read_sql_table(table.name, table.bind, table.schema)
   10 loops, best of 3: 61.8 ms per loop

   In [12]: big_table = odo(
       ...:     pd.DataFrame({
       ...:         'a': np.random.rand(1000000),
       ...:         'b': np.random.rand(1000000)},
       ...:     ),
       ...:     'postgresql://localhost/test::largefloattest',
       ...: )

   In [13]: %timeit warp_prism.to_dataframe(big_table)
   1 loop, best of 3: 248 ms per loop

   In [14]: %timeit odo(big_table, pd.DataFrame)
   1 loop, best of 3: 1.51 s per loop

   In [15]: %timeit pd.read_sql_table(big_table.name, big_table.bind)
   1 loop, best of 3: 1.9 s per loop


Installation
------------

Warp Prism can be pip installed but requires numpy to build its C extensions:

.. code-block::

   $ pip install numpy
   $ pip install warp_prism


License
-------

Warp Prism is licensed under the Apache 2.0.

Warp Prism is sponsored by `Quantopian <https://www.quantopian.com>`_ where it
is used to fetch data for use in `Zipline <http://www.zipline.io/>`_ through the
`Pipeline API <https://www.quantopian.com/tutorials/pipeline>`_ or interactively
with `Blaze <http://blaze.readthedocs.io/en/latest/index.html>`_.

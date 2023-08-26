###############
Operations
###############

We provide various basic operations allowing users to modify the built database for ML applications.
These operations include building database from different data providers;aggregating datasets from diverse source;
splitting datasets to training/test set;sanity check/filtering scenarios.
All commands can be run with ``python -m scenarionet.[command]``.
The parameters for each script can be found by adding a ``-h`` flag.

.. note::
    When running ``python -m``, make sure the directory you are at doesn't contain a folder called ``scenarionet``.
    Otherwise, the running may fail.

List
~~~~~

This command can list all operations with detailed descriptions::

    python -m scenarionet.list


Convert
~~~~~~~~

**ScenarioNet doesn't provide any data.**
Instead, it provides converters to parse common open-sourced driving datasets to an internal scenario description, which comprises scenario databases.
Thus converting scenarios to our internal scenario description is the first step to build the databases.
Currently,




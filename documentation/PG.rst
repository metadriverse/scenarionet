############
PG
############

| Website: https://metadriverse.github.io/metadrive/
| Download: *N/A (collected online)*
| Paper: https://arxiv.org/pdf/2109.12674.pdf

The PG scenarios are collected by running simulation and record the episodes in MetaDrive simulator.
The name PG refers to Procedural Generation, which is a technique used to generate maps.
When a map is determined, the vehicles and objects will be spawned and actuated  according to a hand-crafted rules.

Build PG Database
===================

If MetaDrive is installed, there is no any further steps required to build the database. Just run the following
command to generate, i.e. 1000 scenarios::

    python -m scenarionet.convert_pg -d /path/to/pg_database --num_scenarios 1000


Known Issues: PG
==================

N/A

#############################
Argoverse 2.0
#############################

| Website: https://www.argoverse.org/index.html
| Download: https://www.argoverse.org/av2.html#download-link


Argoverse 2 is a collection of open-source autonomous driving data and high-definition (HD) maps from six U.S. cities: Austin, Detroit, Miami, Pittsburgh, Palo Alto, and Washington, D.C. This release builds upon the initial launch of Argoverse (“Argoverse 1”), which was among the first data releases of its kind to include HD maps for machine learning and computer vision research.

Argoverse 2 Motion Forecasting Dataset: contains 250,000 scenarios with trajectory data for many object types. This dataset improves upon the Argoverse 1 Motion Forecasting Dataset.


1. Install av2
==========================

First of all, we have to install the ``av2`` package.

You can following the instructions here: https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data

2. Download Data
===========================

You can following the instructions here: https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data

3. Build av2 Database
============================

    python -m scenarionet.convert_argoverse2 -d /path/to/your/database --raw_data_path /path/to/your/raw_data

Known Issues: Argoverse2
======================

N/A

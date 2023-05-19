# ScenarioNet

ScenarioNet: Scalable Traffic Scenario Management System for Autonomous Driving

## Installation

```
git clone git@github.com:metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

## Usage

We provide some explanation and demo for all scripts here.
You are encouraged to them on your own, add ```-h``` or ```--help``` argument to know more details about these scripts.

### Convert

**Waymo**: the following script can convert Waymo tfrecord to Metadrive scenario description and
store them at directory ./waymo

```
python -m scenarionet.convert_waymo -d waymo --raw_data_path /path/to/tfrecords --num_workers=16
```

**nuPlan**: the following script will convert nuPlan split containing .db files to Metadrive scenario description and
store them at directory ./nuplan

```
python -m scenarionet.convert_nuplan -d nuplan -raw_data_path /path/to/dir/containing/.db files --num_workers=16
```

**nuScenes**: as nuScenes split can be read by specifying version like v1.0-mini and v1.0-training, the following script
will convert all scenarios in that split

```
python -m scenarionet.convert_nuscenes -d nuscenes --version v1.0-mini --num_workers=16
```

**PG**: the following script can generate 10000 scenarios stored at directory ./pg

```
python -m scenarionet.scripts.convert_pg -d pg --num_workers=16 --num_scenarios=10000
```

### Merge & move

For merging two or more database, use
```python -m scenarionet.merge_database -d /destination/path --from_databases /database/path1 /database/path2 ... ```
As a database contains a path mapping, one should move database folder with the following script instead of ```cp```
command
```python -m scenarionet.move_database --to /destination/path --from /source/path```

### Verify

```python -m scenarionet.verify_simulation```
```python -m scenarionet.verify_completeness```
```python -m scenarionet.verify_completeness```
```python -m scenarionet.generate_from_error_file```

### visualization

Visualizing the simulated scenario
```python -m scenarionet.run_simulation```


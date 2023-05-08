#!/usr/bin/env bash

nohup python _test_convert_nuplan.py > nuplan.log 2>&1 &
nohup python _test_convert_nuscenes.py > nuscenes.log 2>&1 &
nohup python _test_convert_pg.py > pg.log 2>&1 &
nohup python _test_convert_waymo.py > waymo.log 2>&1 &
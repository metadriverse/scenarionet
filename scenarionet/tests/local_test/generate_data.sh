#!/usr/bin/env bash

nohup python _test_convert_nuplan.py 2>&1 &
nohup python _test_convert_nuscenes.py 2>&1 &
nohup python _test_convert_pg.py 2>&1 &
nohup python _test_convert_waymo.py 2>&1 &
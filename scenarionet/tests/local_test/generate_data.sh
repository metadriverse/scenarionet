#!/usr/bin/env bash

nohup python ../../convert_nuplan.py --overwrite --test > nuplan.log 2>&1 &
nohup python ../../convert_nuscenes.py --overwrite > nuscenes.log 2>&1 &
nohup python ../../convert_pg.py --overwrite  > pg.log 2>&1 &
nohup python ../../convert_waymo.py --overwrite > waymo.log 2>&1 &
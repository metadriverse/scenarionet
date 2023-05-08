#!/usr/bin/env bash

nohup python ../../scripts/convert_nuplan.py --overwrite > nuplan.log 2>&1 &
nohup python ../../scripts/convert_nuscenes.py --overwrite > nuscenes.log 2>&1 &
nohup python ../../scripts/convert_pg.py --overwrite  > pg.log 2>&1 &
nohup python ../../scripts/convert_waymo.py --overwrite > waymo.log 2>&1 &
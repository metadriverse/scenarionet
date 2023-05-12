python -m scenarionet.scripts.convert_pg -n pg -d pg_0 --start_index=0 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_1 --start_index=5000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_2 --start_index=10000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_3 --start_index=15000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_4 --start_index=20000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_5 --start_index=25000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_6 --start_index=30000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_7 --start_index=35000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_8 --start_index=40000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.convert_pg -n pg -d pg_9 --start_index=45000 --num_workers=20 --num_scenarios=5000
python -m scenarionet.scripts.combine_dataset dataset_path ./ --from_datasets pg_0 pg_1 pg_2 pg_3 pg_4 pg_5 pg_6 pg_7 pg_8 pg_9
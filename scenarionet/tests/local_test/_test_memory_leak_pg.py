import os

from metadrive.constants import DATA_VERSION

from scenarionet import TMP_PATH
from scenarionet.converter.pg.utils import convert_pg_scenario, get_pg_scenarios
from scenarionet.converter.utils import write_to_directory_single_worker


def _test_pg_memory_leak():
    path = os.path.join(TMP_PATH, "test_memory_leak")
    scenario_indices = get_pg_scenarios(0, 1000)
    write_to_directory_single_worker(
        convert_pg_scenario,
        scenario_indices,
        path,
        DATA_VERSION,
        "pg",
        worker_index=0,
        report_memory_freq=10,
        overwrite=True
    )


if __name__ == '__main__':
    _test_pg_memory_leak()

"""
Simple one-click demo for ScenarioNet / Traffic_automation.

- Creates a small procedural (PG) demo database if it does not exist
- Replays scenarios in MetaDrive using scenarionet.sim

Run from repo root as:
    python examples/simple_demo.py
"""

import sys
import subprocess
from pathlib import Path


def run_cmd(cmd):
    """Run a subprocess command and stream its output."""
    print("\n>> Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    # Repo root = parent of this file's directory
    repo_root = Path(__file__).resolve().parents[1]

    # Where we will store demo data
    demo_root = repo_root / "demo_data"
    demo_db = demo_root / "pg_demo_db"

    demo_root.mkdir(exist_ok=True)

    # 1. Create a small procedural-generation (PG) database if needed
    if not demo_db.exists():
        print(f"Creating demo database at: {demo_db}")

        cmd_convert = [
            sys.executable,
            "-m",
            "scenarionet.convert_pg",
            "--database_path",
            str(demo_db),
            "--dataset_name",
            "pg_demo",
            "--num_scenarios",
            "5",
        ]
        run_cmd(cmd_convert)
    else:
        print(f"Using existing demo database at: {demo_db}")

    # 2. Run simulation using scenarionet.sim (2D render for lighter demo)
    print("\nStarting simulation demo...")
    cmd_sim = [
        sys.executable,
        "-m",
        "scenarionet.sim",
        "--database_path",
        str(demo_db),
        "--render",
        "2D",
    ]
    run_cmd(cmd_sim)

    print("\n✅ Demo finished. You can re-run `python examples/simple_demo.py` any time.")


if __name__ == "__main__":
    main()

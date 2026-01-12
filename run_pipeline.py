import subprocess
import sys
import os


PYTHON = sys.executable  # ensures venv python is used


def run_step(script_path, step_name):
    print(f"\n===== Running: {step_name} =====")

    command = [PYTHON, script_path]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ Error in {step_name}")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)
    print(f"✅ {step_name} completed successfully.")


def main():
    print("\n=== Rossmann Sales Prediction Pipeline Started ===")

    run_step("src/data_preprocessing.py", "Data Preprocessing")
    run_step("src/train_model.py", "Model Training")
    run_step("src/predict.py", "Prediction Test")

    print("\n🎯 Pipeline executed successfully.")
    print("Your sales prediction system is ready.")


if __name__ == "__main__":
    main()

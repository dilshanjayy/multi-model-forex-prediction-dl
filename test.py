import os
import subprocess
from kaggle_secrets import UserSecretsClient

# Configuration
user_secrets = UserSecretsClient()
GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
GITHUB_USER = "your_github_username"
REPO_NAME = "multi-model-forex-prediction-dl"
BRANCH = "dev"

# Paths (Kaggle specific)
WORKING_DIR = f"/kaggle/working/{REPO_NAME}"
repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

# Clean Sync Logic
if not os.path.exists(WORKING_DIR):
    print(f"--- First Time Setup: Cloning {BRANCH} ---")
    subprocess.run(["git", "clone", "-b", BRANCH, repo_url], cwd="/kaggle/working")
    os.chdir(WORKING_DIR)
else:
    os.chdir(WORKING_DIR)
    subprocess.run(["git", "fetch", "origin", BRANCH])
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"])
    subprocess.run(["git", "clean", "-fd"])

set -Eeuo pipefail

# -------- Config --------
PROJECT_PARENT_DIR="/kaggle/"
REPO_SSH_URL="git@github.com:tonyvu1289/multi-model-fact-checking.git"
PROJECT_DIR_NAME="multi-model-fact-checking"
VENV_DIR_NAME=".venv"
INPUT_KAGGLE_DATASET_DIR="/kaggle/input/datasets/congduyvu"
WORKING_BRANCH="train_resume"
# Paste your private key content here if needed. Leave empty to auto-generate a key pair.

# -------- Logging --------
log() {
	echo "[setup] $*"
}

# -------- Common Helpers --------
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

run_as_root() {
	if command_exists sudo; then
		sudo "$@"
	else
		"$@"
	fi
}

# -------- System Setup --------
install_base_packages() {
	if command_exists apt-get; then
		log "Installing base packages with apt..."
		run_as_root apt-get update -y
		run_as_root apt-get install -y git openssh-client python3 python3-venv python3-pip curl unzip pv
	else
		log "apt-get not found. Please install manually: git openssh-client python3 python3-venv python3-pip curl unzip pv"
	fi
}

# -------- GitHub SSH Setup --------

clone_repo() {
	cd "$PROJECT_PARENT_DIR"
	# If the repo already exists, rm -rf and re-clone to ensure we have the latest code and a clean state.
	if [[ -d "$PROJECT_DIR_NAME" ]]; then
		log "Project directory $PROJECT_DIR_NAME already exists. Removing it for a fresh clone."
		rm -rf "$PROJECT_DIR_NAME"
	fi	
	git clone git@github.com:tonyvu1289/multi-model-fact-checking.git
	cd "$PROJECT_DIR_NAME"
	git checkout "$WORKING_BRANCH"
}

set_github_ssh() {
	if [[ -d "$INPUT_KAGGLE_DATASET_DIR/github-ssh" ]]; then
		mkdir -p "$HOME/.ssh"
		cp -r "$INPUT_KAGGLE_DATASET_DIR/github-ssh/.ssh" "$HOME/.ssh/"	
		chmod 600 "$HOME/.ssh/"
		log "GitHub SSH keys copied from $INPUT_KAGGLE_DATASET_DIR/github-ssh to $HOME/.ssh/"
	else 
		log "No GitHub SSH keys found in $INPUT_KAGGLE_DATASET_DIR/github-ssh. Setting up SSH keys."
	fi
}
# -------- Python Environment --------
setup_python_env() {
	if [[ "$INSTALL_PY_DEPS" != "1" ]]; then
		log "Python dependency installation is disabled (INSTALL_PY_DEPS is not 1)."
		return
	fi

	cd "$PROJECT_PARENT_DIR/$PROJECT_DIR_NAME"

	log "Creating virtual environment at $VENV_DIR_NAME"
	python3 -m venv "$VENV_DIR_NAME"

	# shellcheck disable=SC1091
	source "$VENV_DIR_NAME/bin/activate"

	log "Upgrading pip/setuptools/wheel"
	pip install --upgrade pip setuptools wheel

	if [[ -f "requirements.txt" ]]; then
		log "Installing requirements.txt"
		pip install -r requirements.txt
	fi

	if [[ -f "misc/factify/requirements.txt" ]]; then
		log "Installing misc/factify/requirements.txt"
		pip install -r misc/factify/requirements.txt
	fi
# -------- Main --------
main() {
	set_github_ssh
	clone_repo
	install_base_packages
	setup_python_env

	log "Done."
	log "Project directory: $PROJECT_PARENT_DIR/$PROJECT_DIR_NAME"
	log "Activate env with: source $PROJECT_PARENT_DIR/$PROJECT_DIR_NAME/$VENV_DIR_NAME/bin/activate"
}

main "$@"

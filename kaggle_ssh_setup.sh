sour#!/usr/bin/env bash
set -Eeuo pipefail

# -------- Config --------
PROJECT_PARENT_DIR="/kaggle/"
REPO_SSH_URL="git@github.com:tonyvu1289/multi-model-fact-checking.git"
PROJECT_DIR_NAME="multi-model-fact-checking"
VENV_DIR_NAME=".venv"
INSTALL_PY_DEPS="1"
GITHUB_SSH_KEY_COMMENT="$USER@$(hostname)-github"
DATA_DOWNLOAD_URL="http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/latest_dataset/mocheg_with_tweet_2023_03.tar.gz"
DATA_OUTPUT_DIR="$PROJECT_PARENT_DIR/$PROJECT_DIR_NAME/data"
DATA_OUTPUT_NAME=""
FORCE_DOWNLOAD="0"
EXTRACT_DATA_ARCHIVE="1"
DATA_EXTRACT_DIR="$DATA_OUTPUT_DIR"

WORKING_BRANCH="train_resume"
# Paste your private key content here if needed. Leave empty to auto-generate a key pair.
GITHUB_SSH_PRIVATE_KEY=""
DOWNLOADED_DATA_PATH=""

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

# -------- Data Download --------
default_download_name_from_url() {
	local url="$1"
	local name

	name="$(basename "${url%%\?*}")"
	if [[ -z "$name" || "$name" == "/" || "$name" == "." ]]; then
		echo "dataset.tar.gz"
	else
		echo "$name"
	fi
}

download_training_data_from_url() {
	local output_name output_path tmp_path

	if [[ -z "$DATA_DOWNLOAD_URL" ]]; then
		log "DATA_DOWNLOAD_URL is empty. Skipping data download."
		return
	fi

	mkdir -p "$DATA_OUTPUT_DIR"

	output_name="$DATA_OUTPUT_NAME"
	if [[ -z "$output_name" ]]; then
		output_name="$(default_download_name_from_url "$DATA_DOWNLOAD_URL")"
	fi

	output_path="$DATA_OUTPUT_DIR/$output_name"
	tmp_path="$output_path.part"

	if [[ -f "$output_path" && "$FORCE_DOWNLOAD" != "1" ]]; then
		log "Dataset already exists at $output_path. Skipping download."
		log "Set FORCE_DOWNLOAD to 1 in the Config section to re-download."
		DOWNLOADED_DATA_PATH="$output_path"
		return
	fi

	log "Downloading dataset from URL to $output_path"
	if [[ -f "$tmp_path" ]]; then
		log "Resuming interrupted download from $tmp_path"
	else
		log "Starting new download"
	fi
	curl -fL --retry 3 --connect-timeout 30 -C - "$DATA_DOWNLOAD_URL" -o "$tmp_path"

	mv "$tmp_path" "$output_path"
	DOWNLOADED_DATA_PATH="$output_path"
	log "Data download complete: $DOWNLOADED_DATA_PATH"
}

extract_training_data_archive() {
	local archive_path="$1"
	local archive_name stamp_file plain_name archive_size

	if [[ "$EXTRACT_DATA_ARCHIVE" != "1" ]]; then
		log "Archive extraction is disabled (EXTRACT_DATA_ARCHIVE is not 1)."
		return
	fi

	if [[ -z "$archive_path" || ! -f "$archive_path" ]]; then
		log "No downloaded archive file found to extract."
		return
	fi

	mkdir -p "$DATA_EXTRACT_DIR"
	archive_name="$(basename "$archive_path")"
	stamp_file="$DATA_EXTRACT_DIR/.extracted.$archive_name.stamp"

	if [[ -f "$stamp_file" && "$stamp_file" -nt "$archive_path" ]]; then
		log "Archive already extracted and unchanged. Skipping extraction."
		return
	fi

	log "Extracting $archive_path to $DATA_EXTRACT_DIR"
	if command_exists pv; then
		archive_size="$(stat -c%s "$archive_path")"
	else
		archive_size=""
		log "Tip: install pv to see extraction progress and ETA."
	fi

	case "$archive_name" in
		*.tar.gz|*.tgz)
			if command_exists pv && [[ -n "$archive_size" ]]; then
				pv -s "$archive_size" "$archive_path" | tar -xzf - -C "$DATA_EXTRACT_DIR"
			else
				tar -xzf "$archive_path" -C "$DATA_EXTRACT_DIR"
			fi
			;;
		*.tar)
			if command_exists pv && [[ -n "$archive_size" ]]; then
				pv -s "$archive_size" "$archive_path" | tar -xf - -C "$DATA_EXTRACT_DIR"
			else
				tar -xf "$archive_path" -C "$DATA_EXTRACT_DIR"
			fi
			;;
		*.zip)
			log "unzip does not provide a reliable ETA."
			log "Continuing ZIP extraction by skipping already extracted files."
			unzip -n "$archive_path" -d "$DATA_EXTRACT_DIR" >/dev/null
			;;
		*.gz)
			plain_name="${archive_name%.gz}"
			if command_exists pv && [[ -n "$archive_size" ]]; then
				pv -s "$archive_size" "$archive_path" | gunzip -c > "$DATA_EXTRACT_DIR/$plain_name"
			else
				gunzip -c "$archive_path" > "$DATA_EXTRACT_DIR/$plain_name"
			fi
			;;
		*)
			log "Unsupported archive type for auto-extract: $archive_name"
			return
			;;
	esac

	touch "$stamp_file"
	log "Extraction complete: $DATA_EXTRACT_DIR"
}

# -------- GitHub SSH Setup --------
github_ssh_probe_output() {
	ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com 2>&1 || true
}

generate_github_ssh_keypair() {
	local key_path="$1"
	log "Generating a new SSH key pair."
	ssh-keygen -t ed25519 -C "$GITHUB_SSH_KEY_COMMENT" -f "$key_path" -N ""
}

ensure_github_known_host() {
	if ! grep -q "github.com" "$HOME/.ssh/known_hosts" 2>/dev/null; then
		log "Adding github.com to known_hosts"
		ssh-keyscan -H github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
	fi
}

setup_github_ssh() {
	local key_path="$HOME/.ssh/id_ed25519"
	local pub_path="$HOME/.ssh/id_ed25519.pub"

	mkdir -p "$HOME/.ssh"
	chmod 700 "$HOME/.ssh"

	if [[ -n "$GITHUB_SSH_PRIVATE_KEY" ]]; then
		if [[ "$GITHUB_SSH_PRIVATE_KEY" == SHA256:* ]]; then
			log "GITHUB_SSH_PRIVATE_KEY appears to be a fingerprint, not a private key."
			log "Provide full key content starting with -----BEGIN ... PRIVATE KEY-----."
			exit 1
		fi

		local key_content
		key_content="${GITHUB_SSH_PRIVATE_KEY//$'\r'/}"
		if [[ "$key_content" == *"\\n"* ]]; then
			key_content="${key_content//\\n/$'\n'}"
		fi
		if ! grep -q "BEGIN .*PRIVATE KEY" <<< "$key_content"; then
			log "GITHUB_SSH_PRIVATE_KEY is not a valid private key block."
			exit 1
		fi

		log "Writing GitHub private key to ~/.ssh/id_ed25519"
		printf '%s\n' "$key_content" > "$key_path"
	else
		if [[ ! -f "$key_path" ]]; then
			log "No private key provided."
			generate_github_ssh_keypair "$key_path"
		elif ! ssh-keygen -yf "$key_path" >/dev/null 2>&1; then
			local broken_key_backup
			broken_key_backup="$key_path.broken.$(date +%Y%m%d%H%M%S)"
			log "Existing ~/.ssh/id_ed25519 is invalid. Backing up to $broken_key_backup"
			mv "$key_path" "$broken_key_backup"
			rm -f "$pub_path"
			generate_github_ssh_keypair "$key_path"
		fi
	fi
	chmod 600 "$key_path"

	if [[ ! -f "$pub_path" ]]; then
		if ! ssh-keygen -y -f "$key_path" > "$pub_path"; then
			log "Failed to derive public key from $key_path"
			exit 1
		fi
	fi
	chmod 644 "$pub_path"

	ensure_github_known_host

	eval "$(ssh-agent -s)" >/dev/null
	if [[ -f "$key_path" ]]; then
		if ! ssh-add "$key_path" >/dev/null 2>&1; then
			log "Failed to add $key_path to ssh-agent."
			log "If the key has a passphrase, run ssh-add manually and retry."
			exit 1
		fi
	fi

	log "Add this PUBLIC key to GitHub (Settings -> SSH and GPG keys -> New SSH key):"
	echo
	cat "$pub_path"
	echo
	log "GitHub page: https://github.com/settings/keys"
}

wait_for_github_ssh_authorization() {
	local output

	while true; do
		read -r -p "After adding the key on GitHub, press Enter to verify (or type q to quit): " answer
		if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
			log "Stopped by user before GitHub SSH authorization was confirmed."
			exit 1
		fi

		output="$(github_ssh_probe_output)"
		if [[ "$output" == *"successfully authenticated"* ]]; then
			log "GitHub SSH authentication verified."
			break
		fi

		log "GitHub SSH auth not ready yet."
		echo "$output"
		log "Please confirm the public key was added to the correct GitHub account, then retry."
	done
}

github_ssh_auth_is_ready() {
	local output
	output="$(github_ssh_probe_output)"
	if [[ "$output" == *"successfully authenticated"* ]]; then
		return 0
	fi
	return 1
}

clone_repo() {
	cd "$PROJECT_PARENT_DIR"
	# If the repo already exists, rm -rf and re-clone to ensure we have the latest code and a clean state.
	if [[ -d "$PROJECT_DIR_NAME" ]]; then
		log "Project directory $PROJECT_DIR_NAME already exists. Removing it for a fresh clone."
		rm -rf "$PROJECT_DIR_NAME"
	fi	
	git clone https://github.com/tonyvu1289/multi-model-fact-checking.git
	git checkout "$WORKING_BRANCH"
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
}

# -------- Main --------
main() {
	install_base_packages
	clone_repo
	setup_python_env
	# download_training_data_from_url
	# extract_training_data_archive "$DOWNLOADED_DATA_PATH"

	log "Done."
	log "Project directory: $PROJECT_PARENT_DIR/$PROJECT_DIR_NAME"
	if [[ -n "$DOWNLOADED_DATA_PATH" ]]; then
		log "Downloaded data file: $DOWNLOADED_DATA_PATH"
	fi
	if [[ "$EXTRACT_DATA_ARCHIVE" == "1" ]]; then
		log "Extracted data directory: $DATA_EXTRACT_DIR"
	fi
	log "Activate env with: source $PROJECT_PARENT_DIR/$PROJECT_DIR_NAME/$VENV_DIR_NAME/bin/activate"
}

main "$@"

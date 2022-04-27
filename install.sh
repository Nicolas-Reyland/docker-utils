#!/bin/sh
if [ ! -f "$PWD/docker-utils.py" ]; then
    echo "Please cd into the git repo directory"
    exit 1
else
    python3.10 -m pip install -r requirements.txt
    chmod +x "$PWD/docker-utils.py"
    ln -s "$PWD/docker-utils.py" "$HOME/.local/bin/docker-utils"
fi

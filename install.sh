#!/bin/sh
if [ ! -f "$PWD/docker-utils.py" ]; then
    echo "Please cd into the git repo directory"
    exit 1
else
    ln -s "$PWD/docker-utils.py" "$HOME/.local/bin/docker-utils"
fi

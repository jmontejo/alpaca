#!/bin/bash

function setup_conda () {

  echo "[INFO] Activating conda..."

  # The user is supposed to enable conda
  # if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  #     . "$HOME/anaconda3/etc/profile.d/conda.sh"
  # else
  #   . "$HOME/miniconda3/etc/profile.d/conda.sh"
  # fi
  conda activate alpaca

  echo "[INFO] Setting env variables..."
  # Getting the source directory in bash
  # https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
  SOURCE="${BASH_SOURCE[0]}"
  while [ -h "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
  done
  export ALPACA_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"

  # Python related env variables
  export PYTHONPATH=${ALPACA_DIR}:${ALPACA_DIR}/lib/python2.7/site-packages:${PYTHONPATH}
  # pip install --user <package> site.USER_BASE location
  # https://pip.pypa.io/en/stable/user_guide/
  export PYTHONUSERBASE=${ALPACA_DIR}

  export PATH=${ALPACA_DIR}/bin:${PATH}

  echo "[INFO] Finished env setup"
}

setup_conda
echo "[INFO] Finished"

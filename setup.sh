#!/bin/bash

function setup_conda () {

  echo "[INFO] Activating conda..."

  if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
      . "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate alpaca

  echo "[INFO] Setting env variables..."
  #ALPACA_DIR=$(dirname $(readlink -f $0))
  # https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
  SOURCE="${BASH_SOURCE[0]}"
  while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
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

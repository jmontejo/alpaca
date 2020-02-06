#!/bin/bash

function setup_conda () {

  echo "[INFO] Activating conda..."
  . "${HOME}/.miniconda3/etc/profile.d/conda.sh"

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


function setup_root () {

  echo "[INFO] Setup ATLAS software... (will take a while)"
  export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
  source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh > /dev/null


  LCG="LCG_96"
  ARCH="x86_64-centos7-gcc8-opt"

  echo "[INFO] Setup root 6.18.00-${ARCH}..."
  lsetup "root 6.18.00-${ARCH}" > /dev/null

  echo "[INFO] Setup python..."
  lsetup python > /dev/null
  echo "[INFO] Setup python module pathlib2..."
  lsetup "lcgenv -p ${LCG} ${ARCH} pathlib2" > /dev/null

  echo "[INFO] Finished root setup"
}


case "$1" in
   "") 
      echo "Usage : $0 [conda|root]"
      ;;
   conda)
      setup_conda
      ;;
   root)
      setup_root
      ;;
esac

#if [[ $1 == root ]]; then
#    setup_root
#else
#    setup_env
#fi

echo "[INFO] Finished"

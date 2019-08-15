#!/bin/bash

# This script executes all the commands to publish edflow to anaconda.
# After running it, you can install edflow using
#
# >>> conda install -c conda-forge -c jhaux edflow
#
# It is recommended to first publish to pypi and then run this script, as it
# pulls the specified version from pypi.

# Get the version of the package from the commandline arguments
VERSION=$1

# Full docu of what is happening can be found here:
# https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html

# If you need conda-build -> This is how you get it
# conda install conda-build

conda skeleton pypi edflow --version $VERSION
sed -i 's/opencv-python/opencv/g' edflow/meta.yaml
conda-build -c conda-forge --python 3.6 edflow
conda-build -c conda-forge --python 3.7 edflow

conda convert -f --platform all /home/jhaux/miniconda3/envs/edbase/conda-bld/linux-64/edflow-$VERSION-py36h39e3cac_0.tar.bz2 -o outputdir/
conda convert -f --platform all /home/jhaux/miniconda3/envs/edbase/conda-bld/linux-64/edflow-$VERSION-py37h39e3cac_0.tar.bz2 -o outputdir/

# To upload you need the anaconda client
# conda install anaconda-client
anaconda login

for folder in $(ls outputdir); do
        for file in $(ls outputdir/"$folder"); do
                anaconda upload outputdir/$folder/$file
        done
done
# Need to also upload the not converted files!
anaconda upload /home/jhaux/miniconda3/envs/edbase/conda-bld/linux-64/edflow-$VERSION-py36h39e3cac_0.tar.bz2
anaconda upload /home/jhaux/miniconda3/envs/edbase/conda-bld/linux-64/edflow-$VERSION-py37h39e3cac_0.tar.bz2

anaconda logout

# Releasing edflow

When releasing to `pipy` and `conda` simply run the `release.sh` script. For this to succeed you need to know Johannes' login credentials.

You can do this all by hand as well. Always publish to `pipy` first and then to `conda`.

## pipy
> see also `release2pypi.sh`

Install the following:
```
pip install twine
# or
conda install -c conda-forge twine
```

Then run the following:
```
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
```

This will push to pypi.

## conda
> see also `release2conda.sh`

Install the following packages:
```
conda install conda-build
conda install anaconda-client
```

Now run the following commands:
```
VERSION=$(python ../setup.py --version)

conda skeleton pypi edflow --version $VERSION

# the opencv package changed its name
# This is a hack which should be fixed at some point
sed -i 's/opencv-python/opencv/g' edflow/meta.yaml

# Build for python 3.6 and 3.7
conda-build -c conda-forge --python 3.6 edflow
conda-build -c conda-forge --python 3.7 edflow

# Convert the package for all relevant platforms.
# Get the <path/to/XX-package> from the conda-build output
conda convert -f --platform all <path/to/3.6-package>.tar.bz2 -o outputdir/
conda convert -f --platform all <path/to/3.7-package>.tar.bz2 -o outputdir/


# Now push to anaconda

anaconda login

for folder in $(ls outputdir); do
        for file in $(ls outputdir/"$folder"); do
                anaconda upload outputdir/$folder/$file
        done
done
# Need to also upload the not converted files!
anaconda upload <path/to/3.6-package>.tar.bz2
anaconda upload <path/to/3.7-package>.tar.bz2

anaconda logout
```

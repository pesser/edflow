from setuptools import setup

setup(name='edflow',
      version='0.1',
      description='Our code to make videos of realisitc human motion.',
      url='http://bitbucket.com/jhaux/edflow',
      author='Patrick Esser, Johannes Haux, Timo Milbich',
      author_email='{patrick.esser, johannes.haux, timo.milbich}'
                   '@iwr.uni-heidelberg.de',
      license='MIT',
      packages=['edflow'],
      install_requires=[
          'pyyaml',
          'opencv-python',
          'tqdm',
          'Pillow',
          'chainer',
          'numpy',
          'scipy',
          'h5py',
          'scikit-learn',
          'scikit-image'
          ],
      zip_safe=False,
      scripts = ["edflow/edflow", "edflow/edcache"])

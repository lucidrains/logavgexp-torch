from setuptools import setup, find_packages

setup(
  name = 'logavgexp-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.6',
  license='MIT',
  description = 'LogAvgExp - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/logavgexp-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'pytorch',
    'logsumexp'
  ],
  install_requires=[
    'einops>=0.4.1',
    'torch>=1.6',
    'unfoldNd'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

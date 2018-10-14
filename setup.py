from setuptools import setup, find_packages

import os
long_description = 'Ascii DAG for visualization of dataflow'

if os.path.exists('README.md'):
  long_description = open('README.md').read()

setup(
  name = "stackeddag",
  version = "0.2",
  description = 'A visualization tool to show a ascii graph from Graphviz-Dot-file or Tensorflow',
  license = 'MIT',
  url = 'https://github.com/junjihashimoto/py-stacked-dag',
  keywords = 'tensorflow tensor machine-learning graphviz ascii dag ml deep-learning neural-network',
  author = "Junji Hashimoto",
  author_email = "junji.hashimoto@gmail.com",
  packages = find_packages(),
  install_requires = [
    'pydot'
  ],
  scripts=['bin/stackeddag.py']
)

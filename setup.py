from setuptools import setup, find_packages

setup(
  name = "stackeddag",
  version = "0.3.3",
  description = 'A visualization tool to show a ascii graph from Graphviz-Dot-file or Tensorflow',
  long_description = open('README').read(),
  license = 'MIT',
  url = 'https://github.com/junjihashimoto/py-stacked-dag',
  keywords = 'tensorflow tensor machine-learning graphviz ascii dag ml deep-learning neural-network',
  author = "Junji Hashimoto",
  author_email = "junji.hashimoto@gmail.com",
  packages = find_packages(),
  install_requires = [
    'pydot'
  ],
  scripts=['bin/stackeddag']
)

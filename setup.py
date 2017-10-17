from setuptools import setup

setup(name='plotly_eda',
      version='0.1',
      description='pltly focused on EDA',
      url='https://github.com/arrubo/plotly_eda',
      author='arrubo',
      author_email='turanzas.ror@gmail.com',
      license='MIT',
      classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Framework :: Jupyter',
      ],
      packages=['plotly_eda'],
      install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'IPython',
        'colorlover',
        'plotly'],
      zip_safe=False)


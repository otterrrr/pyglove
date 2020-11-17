try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """\
This is pure python version of stanford GloVe word embeddings (https://github.com/stanfordnlp/GloVe)
Most of this work was to port pure C code to pure python code so it has almost same logic as that of original GloVe implementation
However for simplicity, this pyglove doesn't concern of memory-aware execution so your system should have enough memory to load your own corpus and intermediate results
For your information, parallelization here is based on python.multiprocessing but much slower than native C implementation using threads
"""

setup(
    name='pyglove',
    version='0.1.0',
    description=('Pure python implementation of GloVe word embeddings'),
    long_description='',
    py_modules=['pyglove'],
    install_requires=['numpy'],
    author='Taesik Yoon',
    author_email='taesik.yoon.02@gmail.com',
    url='https://github.com/otterrrr/pyglove',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT license",
        "Topic :: Scientific/Engineering :: Natural Language Processing",
        "Operating System :: OS Independent"
    ]
)

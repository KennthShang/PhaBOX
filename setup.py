from setuptools import setup, find_packages

setup(
    name='phabox',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'prodigal',
        'pprodigal',
        'biopython',
        'pytorch',
        'networkx',
        'transformers',
        'pyarrow',
    ],
    author='SHANG JIAYU',
    author_email='jiayushang@cuhk.edu.hk',
    description='Local version of the phage identification and analysis web server',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KennthShang/PhaBOX',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

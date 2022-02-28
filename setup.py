from setuptools import setup, find_packages

from torchcast import __version__

setup(
    name='torchcast',
    version=__version__,
    license='MIT',
    packages=find_packages(include='torchcast.*'),
    zip_safe=False,
    install_requires=[
        'backports.cached-property',
        'torch>=1.8',
        'numpy>=1.4'
    ],
    extras_require={
        'tests': ['parameterized>=0.7', 'filterpy>=1.4', 'pandas>=1.0'],
        'docs': [
            'jupytext>=1.11',
            'plotnine>=0.8',
            'nbsphinx>=0.8.2',
            'ipykernel>=5.3.4',
            'tqdm>=4.59',
            'ipywidgets>=7.6.3',
            'sphinx_rtd_theme>=0.5.2',
            'pandoc>=1.0.2',
            'pytorch_lightning>=1.5',
            'torch_optimizer>=0.3.0',
            'matplotlib'
        ]
    }
)

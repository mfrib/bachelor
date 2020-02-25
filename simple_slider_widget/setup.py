"""
Setup file for package `simple_slider`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'simple_slider'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='plotting results in a widget',
        version='0.0.1',
        long_description=(HERE / "Readme.md").read_text(),
        long_description_content_type='text/markdown',
        url='til-birnstiel.de',
        author='Til Birnstiel',
        author_email='til.birnstiel@lmu.de',
        license='GPLv3',
        packages=[PACKAGENAME],
        package_dir={PACKAGENAME: PACKAGENAME},
        package_data={PACKAGENAME: ['data.csv']},
        install_requires=[
            'numpy',
            'matplotlib',
            'astropy'],
        zip_safe=False,
        entry_points={'console_scripts': ['simple_slider=simple_slider.__init__:main']}
        )

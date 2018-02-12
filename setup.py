from setuptools import setup


setup(
    name='pydense2',
    version='0.1',
    description='A pythonic wrapper to the PyDenseCRF package',
    long_description='See https://github.com/yngvem/PyDense2',
    author='Yngve Mardal Moe',
    author_email='yngve.m.moe@gmail.com',
    url='https://github.com/yngvem/PyDense2',
    packages=['pydense2'],
    setup_requires=['numpy', 'pydensecrf'],
    license='MIT',
)
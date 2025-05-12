from setuptools import setup, find_packages

setup(
    name="multi-agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'openai',
        'pandas',
        'matplotlib',
        'python-dotenv',
    ],
) 
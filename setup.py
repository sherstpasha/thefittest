import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name = 'thefittest',
    packages = ['SelfCEA'],
    version = '1.0.0',
    license='MIT',
    description = 'Implementation of data mining methods that use evolutionary algorithms',
    long_description = long_description,
    long_description_content_type='text/markdown',
    author = 'Pavel Sherstnev',
    author_email = 'sherstpasha99@gmail.com',
    url = 'https://github.com/sherstpasha/thefittest',
    download_url = '',
    keywords = ['genetic algorithm', 'machine learning', "optimization"],
    install_requires = ['numpy'],
    classifiers = ['Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10'])

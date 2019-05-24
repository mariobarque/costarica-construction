from setuptools import setup

install_requires = [str(ir.req) for ir in reqs]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='costarica-construction',
      version='1.0.0',
      description='costarica-construction',
      long_description=readme(),
      classifiers=[

      ],
      keywords='',
      url='',
      author='',
      author_email='',
      license='Proprietary',
      packages=setup.find_packages(),
      install_requires=install_requires,
      )

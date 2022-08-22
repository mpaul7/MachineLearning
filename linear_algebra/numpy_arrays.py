
import click
from numpy import array, empty, zeros, ones, vstack, hstack

@click.group()
def cli():
    """ Mapping and comparing labels"""
    pass

@cli.command(name='ndarray')
def ndarray_create():
    emp = empty([3, 3])
    print(emp)

    _zero = zeros([3,5])
    print(_zero)

    _ones = ones(5)
    print(_ones)

    a1 = array([1, 2, 3])
    # create second array
    a2 = array([4, 5, 6])
    # create horizontal stack
    _vstack = vstack((a1, a2))
    print(_vstack)
    
    _hstack = hstack((a1, a2))
    print(_hstack)



if __name__ == "__main__":
   cli()
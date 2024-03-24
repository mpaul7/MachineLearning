import os
import click
import glob
import timeit
import subprocess
import pandas as pd
import numpy as np
import hashlib
from process_data import PCAPExtract
from ipaddress import IPv4Interface
from ipaddress import IPv4Network
from ipaddress import IPv4Address
from collections import defaultdict
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)


@click.group()
def cli():
    """ Label flows based on ports (dns, http), 'reverse_ip' and 'sni' """
    pass

@cli.command(name='extract')
@click.argument('pcap', type=click.Path(exists=True))
# @click.argument('output', type=click.Path(exists=True))
def denoise(pcap, output):

    try:
        print(f'Processing file -> {pcap}')
        _head, _tail = os.path.split(pcap)
        """ Extract  SNI, NFS and DNS data """
        df = PCAPExtract().extract_data(pcap) 
        print(df)
        
    except Exception as e:
        """"""
        print(e, 1111)
        pass

    if __name__ == "__main__":
        cli()
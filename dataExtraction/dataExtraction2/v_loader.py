#!/usr/bin/python
'''
Created on Mar 12, 2018

@author: sdolgkih

basic pcap processing and flow extraction with dpkt
'''

from hashlib import md5
import pickle
import cpkt
import os


def load(rdd, use_hdfs=False, use_bytes=False):
    if use_bytes:
        result = cpkt.process_bytes(rdd)
        fsize = len(rdd)
    else:
        # local file
        result = cpkt.process_file(rdd, False)
        fsize = os.path.getsize(rdd)
    flows = result[-1]
    packets = result[0]
    ft = result[6]
    lt = result[7]
    tbytes = result[5]
    nontup = result[1] + result[4]
    frag = result[8]

    for flow in flows:
        flowhash = md5(pickle.dumps((flow[25], flow[26], flow[27], flow[28], flow[17]))).hexdigest()
        flow[-1] = flowhash

    keys = [f[-1] for f in flows]
    addr = [f[25:29] for f in flows]
    pack = [packets, ft/1000000, lt/1000000, fsize, tbytes, nontup, frag]
    return pack, addr, keys, flows

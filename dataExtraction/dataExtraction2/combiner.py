'''
Created on December, 2019

@author: Hongxin
'''

from hashlib import md5
import pickle
import cpkt
import csv
import numpy as np
import os
import glob
import time

all_feats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,
             16, 17, 18, 19, 20, 21, 29, 30, 31, 32, 35, 36, 37, 38]


def run(pcap, labels, out):
	"""
	Process pcap input, create training file with labels from csv mapped by md5 hash
	"""
	print(pcap)
	rdd_path = pcap
	pack, addr, keys, feat = load(rdd_path, use_hdfs=False)

	flow_data = [[x[i] for i in all_feats] for x in feat]

	feat_dict = dict(zip(keys, flow_data))
	lab_path = labels
	key_val = np.genfromtxt(lab_path, delimiter=',', dtype=np.str)
	lab_dict = {k: v for k, v in key_val}

	path = out
	with open(path, 'w') as csv_file:
		writer = csv.writer(csv_file)
		for k in feat_dict:
			if k in lab_dict:
				# v = feat_dict[k][:-2] + [lab_dict[k]]
				v = feat_dict[k] + [lab_dict[k]]
				writer.writerow(v)


BASE_DIR = r'C:\Users\Owner\mltat\Qosmos\data'
LABELS_SUBDIR = 'labels'
PCAPS_SUBDIR = 'pcaps'
COMBINED_SUBDIR = 'features'

application_to_dir = {
	# 'audio_chat': os.path.join(BASE_DIR, 'audio-chat'),
	'audio-stream': os.path.join(BASE_DIR, 'audio-stream'),
	# 'file_transfer': os.path.join(BASE_DIR, 'file-transfer'),
	# 'p2p': os.path.join(BASE_DIR, 'p2p'),
	# 'text_chat': os.path.join(BASE_DIR, 'text-chat'),
	# 'video_chat': os.path.join(BASE_DIR, 'video-chat'),
	# 'video': os.path.join(BASE_DIR, 'video-stream'),
	# 'web': os.path.join(BASE_DIR, 'web-browsing')
}

for application, application_dir in application_to_dir.items():

	found_dirs = False
	parent_paths_list = []
	application_dir = os.path.join(BASE_DIR, application_dir, '*')
	subdirs = glob.glob(application_dir)
	while len(subdirs) > 0:
		for subdir in subdirs:
			if os.path.isdir(subdir) and subdir.rsplit('/', 1)[-1] == LABELS_SUBDIR:
				parent_paths_list.append(application_dir.rsplit('/', 1)[0])
				found_dirs = True
				break

		application_dir = os.path.join(application_dir, '*')
		subdirs = glob.glob(application_dir)

	if not found_dirs:
		print("ERROR: Reached end of tree before finding directories")
		break

	for parent_paths in parent_paths_list:
		for parent_path in glob.glob(parent_paths):
			labels_dir_path = os.path.join(parent_path, LABELS_SUBDIR)
			if os.path.isdir(labels_dir_path):
				output_dir_path = os.path.join(parent_path, COMBINED_SUBDIR)
				if not os.path.isdir(output_dir_path):
					os.mkdir(output_dir_path)

				labels_file_list = (labels_file for labels_file in os.listdir(labels_dir_path) if
				                    os.path.isfile(os.path.join(labels_dir_path, labels_file)))
				for labels_file_name in labels_file_list:
					file_name_base = labels_file_name.rstrip('labels.csv').rstrip('_')
					labels_file_path = os.path.join(labels_dir_path, labels_file_name)
					pcap_file_path = os.path.join(parent_path, PCAPS_SUBDIR, file_name_base + '.pcap')
					if os.path.isfile(pcap_file_path):
						run(pcap_file_path, labels_file_path,
						    os.path.join(parent_path, COMBINED_SUBDIR, file_name_base + '_combined.csv'))
					else:
						print(pcap_file_path)

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

if __name__ == '__main__':
	run(r"C:\Users\Owner\mltat\Qosmos\data\audio-stream\spotify\pcaps\NOTQLEAN_20200610_spotify__ios__wf01.pcap",
	    r"C:\Users\Owner\mltat\Qosmos\data\audio-stream\spotify\labels\NOTQLEAN_20200610_spotify__ios__wf01_label.csv")
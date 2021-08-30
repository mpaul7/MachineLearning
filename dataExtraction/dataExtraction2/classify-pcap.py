import os
import time
import subprocess


def listFiles(paths=None):
    """Lists the name of file in folder(s)

        Parameters:
        paths(list): list of folder(s)

        Returns:
        list: list of file names in folder(s)
       """
    l_files = []
    # r=root, d=directories, f = files
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                if '.pcap' in file:
                    l_files.append(os.path.join(r, file))
    return l_files


def mltat_classify(list_pcaps = None, sleep=None):
    """Classify the pcaps in the list using MLTAT system

        Parameters:
        list_pcaps(list): list of pcap names
       """
    CLASSIFY_COMMAND = ['/vagrant/dist/mltat', 'data', 'classify', '']
    for pcap_file in list_pcaps:
        CLASSIFY_COMMAND[-1] = pcap_file
        print("classifying: {} ".format(pcap_file))
        subprocess.call(CLASSIFY_COMMAND)
        time.sleep(sleep)


if __name__ == '__main__':
    paths = ['/vagrant/test_data/pcaps-1']
    list_pcaps = listFiles(paths=paths)
    print("Total pcaps {}".format(len(list_pcaps)))
    mltat_classify(list_pcaps=list_pcaps, sleep=90)
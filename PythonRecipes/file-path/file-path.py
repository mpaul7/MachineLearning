import hashlib
import ntpath
import os
import pandas as pd

""":parameter
When a module is loaded from a file in Python, __file__ is set to its path. 
You can then use that with other functions to find the directory that the file is located in.
"""
def path_test():
    config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf")
    print(config)
    print(__file__)

def test2(path=None):
    head, tail = ntpath.split(path)
    print(head)
    print(tail)
    print(ntpath.basename(head))

def test2():
    type_dict = {'rf': [{'label': 'ext_clf_name', 'prob': 'ext_clf_prob'}, 'EXT_CLF_SAV_MODEL', 'ALL_FIN_FEATS'],
                 'hgmm': [{'label': 'hgmm_clf_name', 'prob': 'hgmm_clf_prob'}, 'HGMM_CLF_SAV_MODEL', 'ALL_HGMM_FEATS']
                 }
    xx =  type_dict.get('rf')[0]
    print(xx)
    # df_test = df_master[type_dict.get(type)[2]]
    # col_name_dict = type_dict.get(type)[0]

def test3():
    filter_kind = "ExactMatchFilter"
    print(filter_kind.find("El"))
    feat_name = 'dport == 80'
    filter_id = int(hashlib.sha256(feat_name.encode('utf-8')).hexdigest(), 16) % 10 ** 3
    print(filter_id)
    feature = ['aaaa', 'bbbb', 'cccc']
    var = ',\n'.join(map(str, feature))
    print(type(var))
    print(var)

    dict = {'num_flows': 53,
            'cpkt':5.3,
            'dns_ftr':0.02,
            'markov': 0.235}
    # df = pd.DataFrame.from_dict(dict, orient='index')
    df = pd.DataFrame(dict, index=['a', 'b', 'c'])
    # df.transpose()
    print(df)

if __name__ == '__main__':
    # path_test()
    # test(r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\video_chat\messenger\pcaps\facebook_video-10min_4.pcap")
    test3()

import pandas as pd
import click
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

import os
from sklearn import preprocessing

sns.set_style("white")
# mpl.rcParams['figure.dpi'] = 300
feature_group = {'gp-1': ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb'],
                 'gp-2': ['min_itf', 'mean_itf', 'mx_itf', 'sdv_itf', 'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb'],
                 'gp-3': ['tot_dur', 'num_pf', 'tot_bf', 'num_pb', 'tot_bb']
                 }
features_all = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb',
                'min_itf', 'mean_itf', 'mx_itf', 'sdv_itf', 'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb',
                'tot_dur', 'num_pf', 'tot_bf', 'num_pb']
FEATURES = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb',
            'min_itf', 'mean_itf', 'mx_itf', 'sdv_itf', 'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb',
            'tot_dur', 'prot',
            'num_pf', 'tot_bf', 'num_pb', 'tot_bb',
            'tot_plf', 'tot_plb', 'fst_plsf', 'fst_plsb', 'num_plpf', 'num_plpb',
            'ent_pb', 'ent_itb', 'label']  # 31

GROUND_TRUTH = ['label']

SPORT = 'sport'
DPORT = 'dport'
SIP = 'sip'
DIP = 'dip'
PROTO = 'protocol'
TIMESTAMP = 'first_timestamp'

P_COUNT_F = 'num_pf'
P_COUNT_B = 'num_pb'


def listFiles(paths=None):
    files = []
    list_paths = []
    # list_paths r=root, d=directories, f = files
    for path in paths:
        for p, d, f in os.walk(path):
            list_paths.append(p)
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(p, file))
    # files.append(file)
    return files


def data_prep(filename, dns_filter=False):
    data = pd.read_csv(filename, index_col=False, names=FEATURES)

    if dns_filter:
        # data = data.loc[(data['label'] != 'network_service')]
        data = data.loc[(data['label'] != 'network_service') & (data[P_COUNT_F] > 1) & (data[P_COUNT_B] > 1)]
    return data


@click.group()
def cli():
    pass


def data_analysis(data1=None, data2=None, feature=None, label=None, normalized=None, app=None, result_path=None):
    df_mean = data2[
        ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb', 'min_itf', 'mean_itf',
         'mx_itf', 'sdv_itf', 'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb', 'num_pf', 'tot_bf', 'num_pb', 'tot_bb',
         'tot_dur']].mean()

    df =  pd.DataFrame(df_mean).T
    df.to_csv(r"C:\Users\Owner\projects3\Generalization\tasks\task-1\comparison_graph\video\data_mean.csv", index=False, header=False, mode='a')


def dist_plot(data1=None, data2=None, feature=None, label=None, normalized=None, app=None, result_path=None):
    # file1 = listFiles(paths=[file1])[0]
    # file2 = listFiles(paths=[file2])[0]
    res = label + r'/' + app + '_' + feature + '.png'
    # print(res)
    fig_path = os.path.join(result_path, res)
    # print(fig_path)
    # data1 = data_prep(file1, dns_filter=True)
    # data2 = data_prep(file2, dns_filter=True)
    # Import data
    x1 = data1.loc[data1.label == label, feature]
    x2 = data2.loc[data2.label == label, feature]
    df_x1 = pd.DataFrame(x1)
    df_x2 = pd.DataFrame(x2)
    print((df_x1))
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    norm_x1 = scaler.fit_transform(df_x1)
    norm_x2 = scaler.fit_transform(df_x2)
    # norm_x1 = scaler.fit_transform([x1])
    # norm_x2 = scaler.fit_transform([x2])
    # print((norm_x1))
    if normalized is 'T':
        kwargs = dict(hist_kws={'alpha': .8}, kde_kws={'linewidth': 1, 'alpha': 2, 'bw': 0.01})
         # kwargs = dict(hist_kws={'alpha': .8}, kde_kws={'linewidth': 1, 'alpha': 2})
    else:
        kwargs = dict(kde=False)

    # label1 = 'Solana-5G - ' + os.path.basename(file1)
    # label2 = 'Solana-Mobile - ' + os.path.basename(file2)
    label1 = os.path.basename(file1)
    label2 = os.path.basename(file2)

    fig = plt.figure(dpi=150)
    fig.set_canvas(plt.gcf().canvas)
    sns.distplot(norm_x1, color="dodgerblue", label=label1, **kwargs)
    sns.distplot(norm_x2, color="orange", label=label2, **kwargs)
    sns.set(font_scale=0.7)
    plt.legend(fontsize='small', title_fontsize='40')
    plt.title(label + ' - ' + feature)
    plt.savefig(fig_path)


# plt.show()


def test():
    x = dict(a=3, b=4)
    y = {'a': 2, 'b': 5}
    print(x)
    print(y)

def main_dist_plot():
    dict = {
        1: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\audio_chat\messenger",
        2: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\audio_chat\skype",
        3: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\audio_stream\sound-cloud",
        4: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\audio_stream\spotify",
        5: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\file_transfer\dropbox\download",
        6: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\file_transfer\dropbox\upload",
        7: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\file_transfer\google-drive\download",
        8: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\mail\gmail",
        9: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\mail\solana-mail",
        10: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\text_chat\messenger",
        11: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\text_chat\telegram",
        12: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\video_chat\messenger",
        13: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\video_chat\skype",
        14: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\video_stream\netflix",
        15: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged\video_stream\youtube"
    }
    for k, v in dict.items():
        print(v)


if __name__ == "__main__":
    # main_dist_plot()
    features = ['tot_bf', 'fst_plsf', 'mx_pf', 'tot_dur', 'mx_itf', 'tot_plf', 'min_pf', 'min_pb']
    # file1_path = r"D:\DataSets\Solana2020a\audio_stream\spotify\features"
    # file2_path = r"D:\DataSets\Solana2020c\audio_stream\spotify\features"
    # data1.to_csv(r"C:\Users\Owner\projects3\Generalization\tasks\task-1\comparison_graph\video\data1.csv")
    # file1 = listFiles(paths=[file1_path])[0]
    # file2 = listFiles(paths=[file2_path])[0]
    file1 = r"C:\Users\Owner\projects3\Generalization\tasks\task-10\data\dis_comp_b_h\data1\6Class_features_Solana2020a_80per.csv"
    file2 = r"C:\Users\Owner\projects3\Generalization\tasks\task-10\data\dis_comp_b_h\data1\6Class_features_Solana2020a_2019_80-5per.csv"
    print(file1, '\n', file2)
    data1 = data_prep(file1, dns_filter=True)
    data2 = data_prep(file2, dns_filter=True)
    for feat in features:
        dist_plot(
            data1=data1,
            data2=data2,
            feature=feat,
            label='social_media',
            app='text_chat',
            normalized='T',
            result_path=r"C:\Users\Owner\projects3\Generalization\tasks\task-10\data\dis_comp_b_h\data1\results")





    #
    #     data_analysis(
    #
    #         data1=data1,
    #         data2=data2,
    #         feature=feat,
    #         label='video',
    #         app='video_stream',
    #         normalized='T',
    #         result_path=r"C:\Users\Owner\projects3\Generalization\tasks\task-1\comparison_graph")

    # for i in range(1, 21):
    #     print(i)
    #     file1 = listFiles(paths=[file1_path])[i]
    #     file2 = listFiles(paths=[file2_path])[i]
    #     print(file1, '\n', file2)
    #     data1 = data_prep(file1, dns_filter=True)
    #     data2 = data_prep(file2, dns_filter=True)
    #     data_analysis(
    #         data1=data1,
    #         data2=data2,
    #         feature=features,
    #         label='video',
    #         app='video_stream',
    #         normalized='T',
    #         result_path=r"C:\Users\Owner\projects3\Generalization\tasks\task-1\comparison_graph")

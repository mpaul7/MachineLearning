import os
import time
import subprocess

path_dict_5g_august2020 = {
    "audio_chat": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-chat\messenger",
                   r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-chat\skype"],
    "audio": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-stream\sound-cloud",
              r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-stream\spotify"],
    "p2p": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\p2p\bittorrent"],
    "text_chat": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\text-chat\telegram"],
    "video_chat": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\video-chat\messenger"],
    "video": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\video-stream\netflix",
              r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\video-stream\youtube"],
    "web": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\web\chrome",
            r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\web\firefox"],
    "mail": [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\mail\gmail"]
}

path_dict_pcaps = {
    "video_chat": [r"C:\Users\Owner\projects3\mltat\test_data\pcaps"]
}
path_dict_Solana_5G = {
    "audio_chat": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\audio_chat\messenger",
                   r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\audio_chat\skype"],
    # "audio": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\audio_stream\sound-cloud\pcaps",
    #           r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\audio_stream\spotify\pcaps"]#,
    # "file_transfer": [
                      # r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\file_transfer\dropbox\download\pcaps",
                      # r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\file_transfer\dropbox\upload\pcaps"],
                      # r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\file_transfer\google-drive\download\pcaps"]#,
                      # r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\file_transfer\google-drive\upload\pcaps"],
                      #   "p2p": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\p2p\bittorrent\pcaps"]#,
    # "text_chat": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\text_chat\telegram\pcaps"]#,
    # "video_chat": [
    #               r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\video_chat\messenger\pcaps"]#,
    #                r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\video_chat\skype\pcaps"]#,
    # "video": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\video_stream\netflix\pcaps",
    #           r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\video_stream\youtube\pcaps"],
    # "web": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\web\chrome\pcaps",
    #         r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\web\firefox\pcaps"],
    # "mail": [r"C:\Users\Owner\5G\dataCollection-data\Solana-5G\mail\gmail\pcaps"]
}

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
                if '.pcap' or '.csv' in file:
                    l_files.append(os.path.join(r, file))
    return l_files


def feature_file(path=None):
    # print(path)
    pcap = 'pcaps'
    # pcap_path = os.path.join(path, pcap)
    # print(pcap_path)
    pcap_path = os.path.join(path, "pcaps")
    label_path = os.path.join(path, "labels")
    print(pcap_path)
    print(label_path)
    # feature_path = path + r'\features'
    list_pcaps = listFiles(paths=[pcap_path])
    list_labels = listFiles(paths=[label_path])
    # print(list_lab)
    print(list_pcaps)
    mltat_combine(list_pcaps=list_pcaps, list_labels=list_labels)
    #
    # CLASSIFY_COMMAND = ['/vagrant/dist/mltat', 'data', 'combine', '', '', '']
    # for (pcap_file, label_file) in zip(list_pcaps, list_labels):
    #     feature_file = os.path.basename(label_file).replace('label', 'feature')
    #     print(feature_file)
    #     CLASSIFY_COMMAND[-3] = pcap_file
    #     CLASSIFY_COMMAND[-2] = label_file
    #     CLASSIFY_COMMAND[-1] = feature_file
    #     print("Combining: {}, {}".format(pcap_file, label_file))
    #     subprocess.call(CLASSIFY_COMMAND)


def mltat_combine(list_pcaps = None, list_labels = None, output_path=None, sleep=None):
    """Classify the pcaps in the list using MLTAT system

        Parameters:
        list_pcaps(list): list of pcap names
       """
    # print(list_pcaps)
    # print(list_labels)
    CLASSIFY_COMMAND = ['/vagrant/dist/mltat', 'data', 'combine', '', '', '']
    for (pcap_file, label_file) in zip(list_pcaps, list_labels):
        feature_file = os.path.basename(label_file).replace('label', 'feature')
        print(feature_file)
        CLASSIFY_COMMAND[-3] = pcap_file
        CLASSIFY_COMMAND[-2] = label_file
        CLASSIFY_COMMAND[-1] = feature_file
        print("Combining: {}".format(pcap_file))
        subprocess.call(CLASSIFY_COMMAND)
        # time.sleep(sleep)
        feature_file = os.path.join(output_path, r'features\{}'.format(os.path.basename(label_file).replace('label', 'feature')))

    # for pcap_file in list_pcaps:
    #     for label_file in list_labels:
    #         # feature_file = os.path.join(output_path, r'features\{}'.format(os.path.basename(label_file).replace('label', 'feature')))
    #         feature_file = os.path.basename(label_file).replace('label', 'feature')
    #         print(feature_file)
            # CLASSIFY_COMMAND[-3] = pcap_file
            # CLASSIFY_COMMAND[-2] = label_file
            # CLASSIFY_COMMAND[-1] = feature_file
            # print("Combining: {}".format(pcap_file))
            # subprocess.call(CLASSIFY_COMMAND)
            # time.sleep(sleep)


if __name__ == '__main__':
    # paths_pcaps = [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-chat\messenger\pcaps"]
    # path_labels = [r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-chat\messenger\labels"]
    # output_path = r"C:\Users\Owner\5G\dataCollection-data\5g-data-august-2020\audio-chat\messenger\features"
    # list_pcaps = listFiles(paths=paths_pcaps)
    # list_labels  = listFiles(paths=path_labels)
    # print("Total pcaps {}".format(len(list_pcaps)))
    # mltat_combine(list_pcaps=list_pcaps, list_labels=list_labels, output_path=output_path,  sleep=5)

    # ==========================================
    for key, value in path_dict_Solana_5G.items():
        print("{}, {}".format(key, value))
        main_label = key
        # print(main_label)
        for app_path in value:
            # print(app_path)
            feature_file(path=app_path)
            print("===================")
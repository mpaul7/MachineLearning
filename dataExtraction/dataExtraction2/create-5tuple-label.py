import glob
import os
import csv
import warnings

import dpkt
import socket
import pickle
from hashlib import md5
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


tuple_header = ['src', 'dst', 'sport', 'dport', 'proto', 'hash', 'pcapfile', 'pcap_path', 'file_size']
label_header = ['hash', 'label']
main_label = ''
dns_label = 'network_service'
mail_label = 'mail'
failed_pcaps = []
# Replace the following path with correct absolute path
path_dict_QosMos = {
    "audio_chat": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\audio-chat\whatsapp\pcaps"],
    "audio": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\audio-stream\deezer\pcaps",
                     r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\audio-stream\spotify\pcaps" ],
    "file_transfer": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\file-transfer\pcaps"],
    "mail": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\mail\gmail\pcaps",
             r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\mail\yahoomail\pcaps" ],
    "p2p": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\p2p\bittorrent\pcaps",
            r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\p2p\frostwire\pcaps"],
    "social_media": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\text-chat\facebook\pcaps",
                     r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\text-chat\sina-weibo\pcaps",
                     r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\text-chat\whatsapp\pcaps" ],
    "video-chat": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-chat\facebook\pcaps",
                   r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-chat\skype\pcaps",
                   r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-chat\whatsapp\pcaps"],
    "video": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-stream\dailymotion\pcaps",
              r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-stream\youtube\pcaps"],
    "web-browsing": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\web-browsing\chrome\pcaps"],
    "gaming": [r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\gaming\pcaps"]
}

path_dict_Solana2020c = {
    # "audio":
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\audio_stream\deezer\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\audio_stream\spotify\pcaps"]#,
    # "file_transfer":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\drive_download_filtered\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\dropbox_filtered\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\skype\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\Whatsapp_filtered\pcaps"]

    # "video":
    #           [r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\4Class\Solana2020c-70-30\pcaps"]#,
    #           # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\video_stream\youtube\pcaps"]
    "audio_chat":
              [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\whatsapp_voip\pcaps"]#,
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\video_stream\youtube\pcaps"]
    # "p2p":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\p2p\pcaps"]
}

path_dict_Solana2020c = {
    # "audio":
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\audio_stream\deezer\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\audio_stream\spotify\pcaps"]#,
    # "file_transfer":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\drive_download_filtered\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\dropbox_filtered\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\skype\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\file_transfer\Whatsapp_filtered\pcaps"]

    # "video":
    #           [r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\4Class\Solana2020c-70-30\pcaps"]#,
    #           # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\video_stream\youtube\pcaps"]
    "audio_chat":
              [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\whatsapp_voip\pcaps"]#,
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\video_stream\youtube\pcaps"]
    # "p2p":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c\p2p\pcaps"]
}
path_dict_Solana_5G = {
    # "audio":
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\audio_stream\sound-cloud\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\audio_stream\spotify\pcaps"]#,
    # "file_transfer":
    #     [r"C:\Users\Owner\mltat\mltat-data\test\pcaps"]        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\file_transfer\dropbox\upload\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\file_transfer\google-drive\download\pcaps"]#,
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\file_transfer\google-drive\upload\pcaps"]

    # "video":
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\video_stream\netflix\pcaps"]#,
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\video_stream\youtube\pcaps"]
    # "audio_chat":
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\audio_chat\messenger\pcaps"]#,
              # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\audio_chat\skype\pcaps"]
    # "p2p":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\p2p\bittorrent\pcaps"]
    "text_chat":
        [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\text_chat\messenger\pcaps"]
    # "video_chat":
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\video_chat\messenger\pcaps"]
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\video_chat\skype\pcaps"]
    # "web":
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\web\chrome\pcaps"]#
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\web\firefox\pcaps"]
    # "mail":
    #     [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\mail\solana\pcaps"]#
        # [r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana-5G\web\firefox\pcaps"]
}


def listFiles(paths=None):
    files = []
    list_paths = []
    # r=root, d=directories, f = files
    for path in paths:
        for p, d, f in os.walk(path):
            list_paths.append(p)
            for file in f:
                if '.pcap' in file:
                    files.append(os.path.join(p, file))
    return files

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def get_five_tuple(pcap_file):
    packet_num = 1
    tmp_list = []
    pcap_path = os.getcwd()
    pcap_file_size = get_file_size(pcap_file)
    f = open(pcap_file, 'rb')

    pcap = dpkt.pcap.Reader(f)

    for ts, buf in pcap:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except:
            print(packet_num)
        packet_num += 1

        # ignore non-ip packet
        if not isinstance(eth.data, dpkt.ip.IP):
            continue

        ip = eth.data
        src = socket.inet_ntoa(ip.src)
        dst = socket.inet_ntoa(ip.dst)

        if ip.p == 6:
            tcp = ip.data
            try:
                tcp.sport
            except:
                continue

            sport = int(tcp.sport)
            dport = int(tcp.dport)
            proto = 6
            # print("ip.src ==== ", type(ip.src))
            # print(ip.src)
            # print("sport ==== ", type(sport))
            # print(sport)
            # print("ip.dst =====", type(ip.dst))
            # print(ip.dst)
            # print("dport ======", type(dport))
            # print(dport)
            # print(" proto ====", type(proto))
            # print(proto)
            hsht = md5(pickle.dumps((ip.src, sport, ip.dst, dport, proto))).hexdigest()
            if not any(hsht in sublist for sublist in tmp_list):
                tmp_list.append([src, dst, sport, dport, proto, hsht, pcap_file, pcap_path, pcap_file_size])

        # UDP
        if ip.p == 17:
            udp = ip.data

            try:
                udp.sport
            except:
                continue

            sport = int(udp.sport)
            dport = int(udp.dport)
            proto = 17

            hshu = md5(pickle.dumps((ip.src, sport, ip.dst, dport, proto))).hexdigest()
            if not any(hshu in sublist for sublist in tmp_list):
                tmp_list.append([src, dst, sport, dport, proto, hshu, pcap_file, pcap_path, pcap_file_size])

    return tmp_list


def create_5tuple(path, tuples_list):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(tuple_header)
        for line in tuples_list:
            writer.writerow(line)


def create_label(path, tuples_list, ):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(label_header)
        for line in tuples_list:
            # print("{}, {}".format(type(line[2]), line[2]))
            if line[2] == 53 or line[3] == 53:
                list_temp = [line[5], dns_label]
            elif line[2] == 25 or line[3] == 25:
                # list_temp = [line[5], mail_label]
                list_temp = [line[5], main_label]
            else:
                list_temp = [line[5], main_label]

            writer.writerow(list_temp)


def run(root_path):
    print(root_path)
    os.chdir(root_path)
    print(root_path)
    for pcap in glob.glob("*.pcap"):
        # ret = get_five_tuple(pcap)
        try:
            ret = get_five_tuple(pcap)

            tuple_file = os.path.join(os.path.dirname(root_path),
                                      r'5tuples\{}'.format(pcap.replace('.pcap', '_5tuple.csv')))
            if not os.path.exists(os.path.dirname(tuple_file)):
                os.makedirs(os.path.dirname(tuple_file))
            #
            # # This function is for personal purpose, comment it here
            create_5tuple(tuple_file, ret)
            #
            label_file = os.path.join(os.path.dirname(root_path),
                                      r'labels\{}'.format(rreplace(pcap, '.pcap', '_label.csv', 1)))

            if not os.path.exists(os.path.dirname(label_file)):
                os.makedirs(os.path.dirname(label_file))

            create_label(label_file, ret)

            print("{} is processed".format(pcap))
        except:
            failed_pcaps.append(pcap)
            pass


if __name__ == '__main__':
    # list_files  = listFiles(paths=[r'C:\Users\Owner\mltat\Qosmos\pcaps'])
    # print(files)
    for key, value in path_dict_Solana_5G.items():
        print("{}, {}".format(key, value))
        main_label = key
        print(main_label)
        for fpcap in value:
            run(fpcap)
    print(failed_pcaps)
    print(len(failed_pcaps))
from scapy.all import *
import dpkt
from collections import defaultdict
load_layer("tls")
import socket as sc

TLS_HANDSHAKE = 22
tls_record = {}

def tls_parsing(pcap_file=None):
	print("hello")
	counters = defaultdict(int)
	# packets = rdpcap(pcap_file)
	# packets.show()
	counter = 1
	with open(pcap_file, 'rb') as pcap:
		pcap = dpkt.pcap.Reader(pcap)
		for ts, buf in pcap:
			try:
				eth = dpkt.ethernet.Ethernet(buf)
				# print 'pkt: %d' % (pkt_count)
				if not isinstance(eth.data, dpkt.ip.IP):
					continue
				ip = eth.data
				if not isinstance(ip.data, dpkt.tcp.TCP):
					continue
				tcp = ip.data
				if tcp.dport != 443 and tcp.sport != 443:
					continue
				if len(tcp.data) <= 0:
					continue

				if tcp.data[0] != TLS_HANDSHAKE:
					continue
				tls_record['hs_type'] = tcp.data
				tls_record['version'] = tcp.data[1]
				tls_record['length'] = tcp.data[2]
				print(
					f' =======\n{counter} -> {sc.inet_ntoa(ip.src)}:{tcp.sport} , {sc.inet_ntoa(ip.dst)}:{tcp.dport}, {tcp.data[1]}, {tls_record}')
				counter += 1

				# ===================================================
				records = []
				try:
					tls = dpkt.ssl.TLS(tcp.data)
					print(tls)
					print(tls.records.data)
				except :
					pass
				# if len(tls.records) < 1:
				# 	continue
				# handshake = dpkt.ssl.TLSHandshake(tls.records[1].data)
				# client_hello = handshake.data
				# print(client_hello)
				# # client_hello.cipher_suite
				# # print(client_hello.num_ciphersuites)
				# for ext in client_hello.extensions:
				# 	# if TLSExtensionTypes.get(ext.value) == 'server_name':
				# 	print(ext)
				# 	# print(ext.name, ext.data[5:])
				# 	exit(0)
				# try:
				# 	records, bytes_used = dpkt.ssl.tls_multi_factory(tcp.data)
				# except dpkt.ssl.SSL3Exception as e:
				# 	print("e1")
				# 	print(e)
				# 	continue
				# except dpkt.dpkt.NeedData as e:
				# 	print("e2")
				# 	print(e)
				# 	continue
				# if len(records) <= 0:
				# 	continue
				#
				# for record in records:
				# 	if record.type != 22:
				# 		continue
				# 	if len(record.data) == 0:
				# 		continue
				# 	if record.data[0] != 1:
				# 		continue
				# 	try:
				# 		handshake = dpkt.ssl.TLSRecord(buf)
				# 		# print(handshake, 111)
				# 	except dpkt.dpkt.NeedData as e:
				# 		print("e3")
				# 		print(e)
				# 		continue
				#
				# 	if not isinstance(handshake.data, dpkt.ssl.TLSClientHello):
				# 		continue
				# 	counters['client_hellos_total'] += 1
				# 	ch = handshake.data
				# 	if ch.version == dpkt.ssl.SSL3_V:
				# 		counters['SSLv3_clients'] += 1
				# 	elif ch.version == dpkt.ssl.TLS1_V:
				# 		counters['TLSv1_clients'] += 1
				# 	elif ch.version == dpkt.ssl.TLS11_V:
				# 		counters['TLSv1.1_clients'] += 1
				# 	elif ch.version == dpkt.ssl.TLS12_V:
				# 		counters['TLSv1.2_clients'] += 1
				#
				# 	print('ch.session_id.version: %s' % dpkt.ssl.ssl3_versions_str[ch.version])
				# 	print('ch.session_id.len: %d' % len(ch.session_id))
				# 	print('ch.num_ciphersuites: %d' % ch.num_ciphersuites)
				# 	print('ch.num_compression_methods: %d' % ch.num_compression_methods)
				# 	print('ch.compression_methods: %s' % str(ch.compression_methods))


				# if ip.p == dpkt.ip.IP_PROTO_TCP:
				# 	dport = int(tcp.dport)
				# 	sport = int(tcp.sport)
				# 	print(sc.inet_ntoa(ip.src), sport, sc.inet_ntoa(ip.dst), dport, ip.p)
				# 	tls = tcp.data
				# 	xx = tls.data
				# 	print(tls.all_extension_headers)
			except dpkt.ssl.SSL3Exception as e:
				print(e)


if __name__ == '__main__':
	# pcap_path = r"C:\Users\Owner\Dropbox\My PC (DESKTOP-5J2CE2M)\Downloads\smallFlows.pcap"
	pcap_path = r"D:\Data\DataSets\Solana2020a\video_stream\netflix\pcaps\Netflix2_8_3_2020.pcap"
	tls_parsing(pcap_file=pcap_path)

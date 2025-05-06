from scapy.all import Ether, IP, TCP, sendp, get_if_hwaddr
import time

# Update this to the correct IP you want to scan
target_ip = "192.168.37.94"
interface = "wlo1"

# Get the MAC address of your wireless interface
src_mac = get_if_hwaddr(interface)

print(f"Simulating SYN scan on {target_ip} via interface {interface}...")

# Common privileged ports
ports_to_scan = range(20, 1025)

for port in ports_to_scan:
    ether = Ether(src=src_mac)
    ip = IP(dst=target_ip)
    tcp = TCP(dport=port, flags='S')
    packet = ether / ip / tcp
    sendp(packet, iface=interface, verbose=False)
    time.sleep(0.01)

print("Attack simulation complete.")

"""
Script to add or remove artificial network delay using tc netem on Linux.
Supports applying delay to a specific TCP port for distributed inference testing.
Run with sudo on each node.
Also supports adding bandwidth limits separately.
"""

import subprocess
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd):
    """Run shell command and handle errors."""
    try:
        subprocess.check_call(cmd, shell=True)
        logging.info(f"Command succeeded: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd} (error: {e})")
        sys.exit(1)

def add_delay(interface, delay_ms, jitter_ms=0, port=None):
    """Add artificial delay using tc netem, optionally for a specific TCP port."""
    delay_str = f"delay {delay_ms}ms"
    if jitter_ms > 0:
        delay_str += f" {jitter_ms}ms"
    
    # Remove existing qdisc if any
    run_command(f"tc qdisc del dev {interface} root")
    
    # Add root qdisc (use prio to allow filtering)
    run_command(f"tc qdisc add dev {interface} root handle 1: prio")
    
    if port:
        # Add filter for specific TCP port (source or destination)
        run_command(f"tc filter add dev {interface} protocol ip parent 1:0 prio 1 u32 match ip dport {port} 0xffff flowid 1:1")
        run_command(f"tc filter add dev {interface} protocol ip parent 1:0 prio 1 u32 match ip sport {port} 0xffff flowid 1:1")
        # Apply netem delay to filtered traffic (band 1)
        run_command(f"tc qdisc add dev {interface} parent 1:1 handle 10: netem {delay_str}")
        logging.info(f"Added delay: {delay_ms}ms (jitter: {jitter_ms}ms) on {interface} for TCP port {port}")
    else:
        # Apply netem delay to all traffic
        run_command(f"tc qdisc add dev {interface} parent 1:1 handle 10: netem {delay_str}")
        logging.info(f"Added delay: {delay_ms}ms (jitter: {jitter_ms}ms) on {interface} for all traffic")

def add_bandwidth(interface, rate, burst, port=None):
    """Add artificial bandwidth limit using tc tbf, optionally for a specific TCP port."""
    if not rate:
        logging.error("Rate must be specified for bandwidth limiting.")
        sys.exit(1)
    
    burst = burst if burst else "32kb"  # Default burst
    
    # Remove existing qdisc if any
    run_command(f"tc qdisc del dev {interface} root")
    
    # Add root qdisc (use prio to allow filtering)
    run_command(f"tc qdisc add dev {interface} root handle 1: prio")
    
    if port:
        # Add filter for specific TCP port (source or destination)
        run_command(f"tc filter add dev {interface} protocol ip parent 1:0 prio 1 u32 match ip dport {port} 0xffff flowid 1:1")
        run_command(f"tc filter add dev {interface} protocol ip parent 1:0 prio 1 u32 match ip sport {port} 0xffff flowid 1:1")
        # Apply tbf bandwidth limit to filtered traffic (band 1)
        run_command(f"tc qdisc add dev {interface} parent 1:1 handle 10: tbf rate {rate} burst {burst} latency 50ms")
        logging.info(f"Added bandwidth limit: {rate} (burst: {burst}) on {interface} for TCP port {port}")
    else:
        # Apply tbf bandwidth limit to all traffic
        run_command(f"tc qdisc add dev {interface} parent 1:1 handle 10: tbf rate {rate} burst {burst} latency 50ms")
        logging.info(f"Added bandwidth limit: {rate} (burst: {burst}) on {interface} for all traffic")

def remove_delay(interface):
    """Remove artificial delay or bandwidth limit."""
    run_command(f"tc qdisc del dev {interface} root")
    logging.info(f"Removed configuration on {interface}")

def main():
    parser = argparse.ArgumentParser(description="Add/Remove artificial network delay or bandwidth limits using tc.")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface (e.g., eth0, wlan0)")
    parser.add_argument("--delay_ms", type=int, default=100, help="Delay in milliseconds (for --add)")
    parser.add_argument("--jitter_ms", type=int, default=0, help="Jitter variation in milliseconds (for --add)")
    parser.add_argument("--rate", type=str, default=None, help="Bandwidth rate limit (e.g., 100kbit, 1mbit) (for --add-bandwidth)")
    parser.add_argument("--burst", type=str, default=None, help="Burst size (e.g., 32kb) (for --add-bandwidth)")
    parser.add_argument("--port", type=int, default=None, help="TCP port to apply configuration to (e.g., 44444 for RPC)")
    parser.add_argument("--add", action="store_true", help="Add delay")
    parser.add_argument("--add-bandwidth", action="store_true", help="Add bandwidth limit")
    parser.add_argument("--remove", action="store_true", help="Remove configuration")
    
    args = parser.parse_args()
    
    action_count = sum([args.add, args.add_bandwidth, args.remove])
    if action_count > 1:
        logging.error("Specify only one of --add, --add-bandwidth, or --remove.")
        sys.exit(1)
    elif action_count == 0:
        logging.error("Specify --add, --add-bandwidth, or --remove.")
        sys.exit(1)
    
    if args.add:
        add_delay(args.interface, args.delay_ms, args.jitter_ms, args.port)
    elif args.add_bandwidth:
        add_bandwidth(args.interface, args.rate, args.burst, args.port)
    elif args.remove:
        remove_delay(args.interface)

if __name__ == "__main__":
    main()
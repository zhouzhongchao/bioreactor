"""Read-only GIGA selection; never probe unrelated boards by opening ports."""
import re

# Arduino mbed_giga 4.6.0 boards.txt: giga.vid/pid and upload_port entries.
GIGA_USB_IDS = {(0x2341, 0x0266), (0x2341, 0x0366), (0x2341, 0x0466)}


def giga_port_rank(port):
    if (getattr(port, 'vid', None), getattr(port, 'pid', None)) in GIGA_USB_IDS:
        return 0
    text = ' '.join(str(getattr(port, field, '') or '') for field in
                    ('description', 'product', 'hwid'))
    if re.search(r'\bGIGA\b', text, re.IGNORECASE):
        return 1
    return None


def select_giga_port(ports, requested=''):
    ports = list(ports)
    if requested:
        candidates = [p for p in ports if p.device.lower() == requested.strip().lower()
                      and giga_port_rank(p) is not None]
    else:
        candidates = [p for p in ports if giga_port_rank(p) is not None]
        if candidates:
            rank = min(giga_port_rank(p) for p in candidates)
            candidates = [p for p in candidates if giga_port_rank(p) == rank]
    if len(candidates) == 1:
        return candidates[0].device
    if len(candidates) > 1:
        raise ValueError('发现多个GIGA串口，请明确选择：' + ', '.join(p.device for p in candidates))
    raise ValueError('未识别到GIGA串口，不会连接UNO或其他设备。请检查GIGA连接、驱动及RST。')

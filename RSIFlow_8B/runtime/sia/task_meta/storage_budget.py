"""System-disk admission checks; persistent experiment data belongs on /root/data."""
import shutil

SYSTEM_DISK_LIMIT_BYTES = 20_000_000_000
RUNTIME_RESERVE_BYTES = 4_000_000_000


def check_system_disk(reserve_bytes=0):
    usage = shutil.disk_usage('/')
    if usage.used + reserve_bytes > SYSTEM_DISK_LIMIT_BYTES:
        raise RuntimeError(
            'SYSTEM_DISK_BUDGET: used=' + str(usage.used)
            + '; reserve=' + str(reserve_bytes)
            + '; ceiling=' + str(SYSTEM_DISK_LIMIT_BYTES)
            + '; move inactive RSI files to /root/data/RSI_iclr2027 before continuing')
    return {'used_bytes': usage.used, 'reserve_bytes': reserve_bytes,
            'ceiling_bytes': SYSTEM_DISK_LIMIT_BYTES}

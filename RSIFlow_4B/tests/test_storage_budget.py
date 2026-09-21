from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sia.task_meta.storage_budget import check_system_disk, RUNTIME_RESERVE_BYTES


def test_admission_reserves_space_below_twenty_decimal_gb():
    with patch('sia.task_meta.storage_budget.shutil.disk_usage', return_value=SimpleNamespace(used=15_000_000_000)):
        assert check_system_disk(RUNTIME_RESERVE_BYTES)['ceiling_bytes'] == 20_000_000_000


def test_admission_does_not_start_a_call_that_would_cross_limit():
    with patch('sia.task_meta.storage_budget.shutil.disk_usage', return_value=SimpleNamespace(used=17_000_000_000)):
        with pytest.raises(RuntimeError, match='SYSTEM_DISK_BUDGET'):
            check_system_disk(RUNTIME_RESERVE_BYTES)


def test_live_check_detects_external_growth():
    with patch('sia.task_meta.storage_budget.shutil.disk_usage', return_value=SimpleNamespace(used=20_000_000_001)):
        with pytest.raises(RuntimeError, match='SYSTEM_DISK_BUDGET'):
            check_system_disk()

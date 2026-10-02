import subprocess
import sys

import pytest

from financepy.utils.calendar import BusDayAdjustTypes, DateGenRuleTypes
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.schedule import Schedule


@pytest.mark.parametrize("frequency", ["CONTINUOUS", "ZERO", "SIMPLE_INTEREST"])
def test_nonperiodic_schedule_frequency_is_rejected_without_looping(frequency):
    # Isolate the process so a regression cannot hang the test runner.
    code = (
        "from financepy.utils.date import Date; "
        "from financepy.utils.schedule import Schedule; "
        "from financepy.utils.frequency import FrequencyTypes; "
        f"Schedule(Date(1,1,2026),Date(1,1,2027),FrequencyTypes.{frequency})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode != 0
    assert "FinError" in result.stderr
    assert "periodic frequency" in result.stderr


@pytest.mark.parametrize("start,end,intermediate", [
    (Date(28, 2, 2026), Date(31, 8, 2026), Date(31, 5, 2026)),
    (Date(30, 4, 2026), Date(31, 10, 2026), Date(31, 7, 2026)),
])
def test_forward_schedule_honors_end_of_month(start, end, intermediate):
    schedule = Schedule(start, end, FrequencyTypes.QUARTERLY,
                        bd_type=BusDayAdjustTypes.NONE,
                        dg_type=DateGenRuleTypes.FORWARD, end_of_month=True)
    assert schedule.schedule_dts() == [start, intermediate, end]

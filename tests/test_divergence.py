"""A run that stops short of its STOP time must not pass as a finished one.

Up to RAMSES 3.84 the engine ended a run on "Divergence. Simulation stopped"
(Newton failed at the minimum time step) by setting its end-of-simulation flag
and nothing else, so ``ramses()`` returned 0, the same flag as a run that
reached STOP. ``execSim()`` read that 0 as success: a reduced GB network
diverged at t = 0.35 s and its notebook reported it as running for three
months, because the trajectory simply ended there.

Later engines return -1 with the message in ``getLastErr()``, which the
non-zero-flag path already raises on. Older engines, and any other early end
the engine does not flag, are caught by comparing the simulated time with the
STOP time. The engine tests below accept either route; the fake-engine tests
pin the STOP-time check on its own, whatever library is bundled.

The diverging case below is the Nordic test system with a Newton tolerance no
iteration can meet, so the very first time step fails at the minimum step size.
That is a reliable way to reach the engine's divergence branch, which is what
is being tested, without shipping a case that diverges for physical reasons.

Every run happens in its own subprocess: RAMSES keeps its state in Fortran
module variables shared by every sim() in a process (see test_live_engine.py).
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import stepss

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "tests" / "data" / "nordic"

STOP = 3.0

FAULT_CLEARED = """  0.000 CONTINUE SOLVER BD 0.020 0.001 0. ABL
  1.000 FAULT BUS 4032 0. 0.8
  1.100 CLEAR BUS 4032
  3.000 STOP
"""

UNREACHABLE_TOLERANCE = "$NEWTON_TOLER 1e-15 1e-15 1e-15 ;\n"

DRIVER = textwrap.dedent(
    """
    import json
    import os
    import sys
    import stepss
    from stepss.globals import RAMSESError

    mode = sys.argv[1]
    case = stepss.cfg()
    for name in ("dyn_A.dat", "volt_rat_A.dat", "settings1.dat"):
        case.addData(name)
    if os.path.exists("extra.dat"):
        case.addData("extra.dat")
    case.addObs("obs.dat")
    case.addDst("case.dst")
    case.addTrj("obs.trj")
    case.addOut("output.trace")

    ram = stepss.sim()
    outcome = {"raised": None, "returned": None}
    try:
        if mode == "run":
            outcome["returned"] = ram.execSim(case)
        else:  # pause part-way, then end the run on purpose
            ram.execSim(case, 1.0)
            outcome["returned"] = ram.endSim()
    except RAMSESError as exc:
        outcome["raised"] = str(exc)
    outcome["sim_time"] = ram.getSimTime()
    json.dump(outcome, open("outcome.json", "w"))
    """
)


def run_case(tmp_path, dst_text, extra=None, mode="run"):
    """Run the Nordic case with *dst_text* in a fresh process; return its outcome."""
    for src in sorted(CASE_DIR.glob("*.dat")):
        shutil.copy(src, tmp_path / src.name)
    (tmp_path / "case.dst").write_text(dst_text)
    if extra is not None:
        (tmp_path / "extra.dat").write_text(extra)
    (tmp_path / "driver.py").write_text(DRIVER)

    # Import the stepss this session imported (the wheel under CI), as
    # test_live_engine.py does.
    import stepss

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(stepss.__file__).resolve().parents[1])]
        + ([env["PYTHONPATH"]] if "PYTHONPATH" in env else []))
    done = subprocess.run([sys.executable, "driver.py", mode], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads((tmp_path / "outcome.json").read_text())


# -- against the engine ---------------------------------------------------

def test_a_diverging_run_raises(tmp_path):
    outcome = run_case(tmp_path, FAULT_CLEARED, extra=UNREACHABLE_TOLERANCE)
    assert outcome["sim_time"] < STOP  # the engine really did stop short
    assert outcome["raised"] is not None, outcome
    assert "Divergence" in outcome["raised"]  # the engine's own words
    # Either the engine flagged it (-1) or the STOP-time check caught it.
    assert ("flag -1" in outcome["raised"]
            or "before the STOP time 3.0000 s" in outcome["raised"])


def test_a_run_that_reaches_stop_does_not_raise(tmp_path):
    outcome = run_case(tmp_path, FAULT_CLEARED)
    assert outcome["raised"] is None, outcome["raised"]
    assert outcome["returned"] == 0
    assert outcome["sim_time"] == pytest.approx(STOP)


def test_ending_a_paused_run_on_purpose_does_not_raise(tmp_path):
    """endSim() stops before STOP by design; that is not a failure."""
    outcome = run_case(tmp_path, FAULT_CLEARED, mode="end")
    assert outcome["raised"] is None, outcome["raised"]
    assert outcome["sim_time"] < STOP


# -- reading the STOP time ------------------------------------------------

def stop_time(path):
    from stepss.cases import _dstStopTime
    return _dstStopTime(path)


def write(tmp_path, text):
    target = tmp_path / "t.dst"
    target.write_text(text)
    return str(target)


def test_the_stop_time_is_read(tmp_path):
    assert stop_time(write(tmp_path, FAULT_CLEARED)) == 3.0


def test_the_first_stop_wins(tmp_path):
    # The engine stops reading at the first STOP, as _dstProblem assumes.
    text = "  1.0 FAULT BUS 4032 0. 0.8\n 9.0 STOP\n20.0 STOP\n"
    assert stop_time(write(tmp_path, text)) == 9.0


def test_comments_are_skipped(tmp_path):
    text = "# 1.0 STOP\n! 2.0 STOP\n\n 5.5 STOP ;\n"
    assert stop_time(write(tmp_path, text)) == 5.5


def test_no_usable_stop_gives_none(tmp_path):
    assert stop_time(write(tmp_path, "  1.0 FAULT BUS 4032 0. 0.8\n")) is None
    assert stop_time(write(tmp_path, "abc STOP\n")) is None
    assert stop_time(str(tmp_path / "absent.dst")) is None


# -- the STOP-time check, against a fake engine ---------------------------

class FakeEngine:
    """Stands in for the loaded library: returns *flag*, stops at *t_end*."""

    def __init__(self, flag, t_end, trace_text=""):
        self.flag, self.t_end, self.trace_text = flag, t_end, trace_text
        self.trace = None

    def set_pause_time(self, t):
        return 0

    def ramses(self, cmd, out):
        self.trace = out.decode("utf-8")
        with open(self.trace, "a") as trace:
            trace.write(self.trace_text)
        return self.flag

    def continue_simul(self):
        return self.flag

    def set_end_simul(self):
        return 0

    def get_sim_time(self):
        return self.t_end

    def get_huge_double(self):
        return 1e308


def faked(tmp_path, engine):
    import stepss

    (tmp_path / "d.dat").write_text("# data\n")
    (tmp_path / "c.dst").write_text(FAULT_CLEARED)
    case = stepss.cfg()
    case.addData(str(tmp_path / "d.dat"))
    case.addDst(str(tmp_path / "c.dst"))
    case.addOut(str(tmp_path / "output.trace"))
    return FakeSim(engine), case


class FakeSim(stepss.sim):
    """A sim that never loads the real library.

    It used to be a real stepss.sim() with its library swapped out, which
    held a reference to the RAMSES DLL for as long as the object lived. A
    test that raises keeps the object alive through the exception's
    traceback until the garbage collector gets to it, and on Windows the
    DLL is only unloaded, and its Fortran state reset, when the last
    reference goes. test_ssa_engine.py relies on that reset between
    fixtures, and on windows-latest with Python 3.12 it once found the
    previous test's simulation still at t = 0.5 s, the other five legs
    passing. A late collection of such an object is the likely cause,
    which is also why it is intermittent; a fake that never loads the
    library cannot cause it either way.
    """

    def __init__(self, engine):  # deliberately not calling sim.__init__
        self._ramseslib = engine
        self._stopTime = None
        self._traceFile = None

    def __del__(self):  # nothing loaded, nothing to unload
        pass


def test_an_unflagged_early_end_raises_with_the_engine_line(tmp_path):
    from stepss.globals import RAMSESError

    engine = FakeEngine(0, 0.35, "t =    0.3500   Divergence. Simulation stopped\n")
    ram, case = faked(tmp_path, engine)
    with pytest.raises(RAMSESError) as err:
        ram.execSim(case)
    assert "t = 0.3500 s, before the STOP time 3.0000 s" in str(err.value)
    assert "t = 0.3500 Divergence. Simulation stopped" in str(err.value)


def test_only_this_calls_part_of_the_trace_is_quoted(tmp_path):
    from stepss.globals import RAMSESError

    (tmp_path / "output.trace").write_text("t =    9.9999   an earlier run's line\n")
    ram, case = faked(tmp_path, FakeEngine(0, 0.35))
    with pytest.raises(RAMSESError) as err:
        ram.execSim(case)
    assert "earlier run" not in str(err.value)


def test_reaching_stop_passes(tmp_path):
    ram, case = faked(tmp_path, FakeEngine(0, 3.0))
    assert ram.execSim(case) == 0


def test_a_latency_early_stop_passes(tmp_path):
    engine = FakeEngine(0, 1.2, "All equipment is latent. So, the simulation stops.\n")
    ram, case = faked(tmp_path, engine)
    assert ram.execSim(case) == 0


def test_a_pause_is_not_checked(tmp_path):
    ram, case = faked(tmp_path, FakeEngine(112, 1.0))
    assert ram.execSim(case, pause=1.0) == 0


def test_a_stop_added_at_run_time_moves_the_check(tmp_path):
    engine = FakeEngine(112, 1.0)
    ram, case = faked(tmp_path, engine)
    ram.execSim(case, pause=1.0)
    engine.add_disturb = lambda t, d: 0
    ram.addDisturb(2.0, "STOP")
    engine.flag, engine.t_end = 0, 2.0
    assert ram.contSim(ram.getInfTime()) == 0

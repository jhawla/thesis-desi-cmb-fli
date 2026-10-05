"""Runs are written on scratch and kept on CFS (``desi_cmb_fli.storage``)."""

from desi_cmb_fli.storage import keep_run


def test_keep_run_copies_then_updates_and_leaves_the_kept_copy_alone(tmp_path):
    run = tmp_path / "scratch" / "run_x"
    (run / "config").mkdir(parents=True)
    (run / "config" / "samples_batch_0.npz").write_bytes(b"a")
    kept = tmp_path / "cfs" / "runs"
    assert keep_run(run, kept) == kept / "run_x"
    assert (kept / "run_x" / "config" / "samples_batch_0.npz").read_bytes() == b"a"
    (run / "figures").mkdir()
    (run / "figures" / "corner.png").write_bytes(b"b")
    keep_run(run, kept)
    assert (kept / "run_x" / "figures" / "corner.png").read_bytes() == b"b"
    assert keep_run(kept / "run_x", kept) is None


def test_keep_run_reports_a_failure_without_raising(tmp_path, capsys):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert keep_run(tmp_path, blocked / "runs") is None
    assert "WARNING" in capsys.readouterr().out

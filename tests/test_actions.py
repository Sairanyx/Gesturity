""" Here we test that actions run when they have to and interface is enforced"""

import pytest

from gesture_unlock.actions import Action, PrintAction


class SpyAction(Action):
    """A fake action that saves if action was run for testing purposes"""

    def __init__(self):
        self.was_run = False

    def run(self) -> None:
        self.was_run = True

def test_spy_action_records_run():
    spy = SpyAction()
    assert spy.was_run is False
    spy.run()
    assert spy.was_run is True


def test_print_action_runs(capsys):
    action = PrintAction("hello")
    action.run()
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_action_interface_is_enforced():
    # An action missing run() cannot even be created.
    class BrokenAction(Action):
        pass
    with pytest.raises(TypeError):
        BrokenAction()

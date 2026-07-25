"""Here actions are triggered when a gesture sequence completes, of course more subclasses could be added
The Recognition code will fire a general action and each action decides what to do. This method will
keep the what happend separate from the what to do about it"""

from abc import ABC, abstractmethod

class Action(ABC):

    """The shared shape is that every action has a run() method included"""

    @abstractmethod
    def run(self) -> None:
        """Idea is that this does the action and the subclasses must implement this"""
        raise NotImplementedError

class PrintAction(Action):
    """Just a print one here for testing  or future messages on sequence reading"""
    def __init__(self, message: str = "Gesture accepted"):
        self._message = message

    def run(self) -> None:
        print(self._message)

class SoundAction(Action):
    """Here is the system sound on completed successful sequence, you can pick your own cool one of course"""

    def run(self) -> None:
        import winsound
        winsound.Beep(800, 200) # Setting the frequency to 880Hz and the duration to 200 ms so short



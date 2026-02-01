from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event, Lock
from collections import defaultdict
import numpy as np
import time


class Spacemouse(Thread):
    def __init__(self, max_value=500, deadzone=(0,0,0,0,0,0), dtype=np.float32):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0

        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        # Event-based motion detection: set when any motion above deadzone occurs
        self._motion_detected = Event()
        self._motion_lock = Lock()

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation, 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def has_motion_occurred(self):
        """Check if any motion above deadzone has occurred since last check.

        This method is useful for detecting intervention triggers during blocking
        operations (like arm_rate.sleep). Unlike get_motion_state() which only
        returns the instantaneous state, this returns True if ANY motion occurred
        since the last call.

        Returns:
            bool: True if motion was detected, False otherwise
        """
        with self._motion_lock:
            occurred = self._motion_detected.is_set()
            self._motion_detected.clear()
            return occurred

    def clear_motion_flag(self):
        """Clear the motion detected flag without checking it."""
        with self._motion_lock:
            self._motion_detected.clear()

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                    # Check if motion is above deadzone and set flag
                    state = np.array(event.translation + event.rotation,
                        dtype=self.dtype) / self.max_value
                    if np.any(np.abs(state) >= self.deadzone):
                        self._motion_detected.set()
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1/1000)
        finally:
            spnav_close()


def test():
    with Spacemouse(deadzone=0.3) as sm:
        for i in range(9999999):
            # print(sm.get_motion_state())
            print(sm.get_motion_state_transformed())
            print(sm.is_button_pressed(0))
            time.sleep(1/100)

if __name__ == '__main__':
    test()

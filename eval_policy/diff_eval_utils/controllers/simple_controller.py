from operator import index
from diff_eval_utils.controllers.base_controller import RobotController
import time
import numpy as np

class SimpleSequentialController(RobotController):
    """Simple controller that executes actions sequentially without buffering."""

    def run(self, n_steps: int):
        """Execute simple sequential control.

        Each iteration:
        1. Get observation
        2. Query policy for actions
        3. Execute all actions sequentially
        """
        n_steps_done = 0
        first=True

        print("Starting simple sequential control...")
        while n_steps_done < n_steps:
            iter_start = time.time()

            # Get observation and action
            self._update_buffer()
            obs_dict = self._get_observation()
            t_inference_start = time.time()
            actions = self.policy_client.predict_action(obs_dict)
            print(f"inference_took {(time.time() - t_inference_start)*1000:.1f} ms")
            print(obs_dict['robot0_eef_pos'])
            # Execute first 8 actions
            t_start = time.time()
            for index, action in enumerate(actions[:8]):
                if n_steps_done >= n_steps:
                    break
                # print(action)
                print(f"action {index} took {(time.time() - t_start)*1000:.1f} ms")
                t_start = time.time()
                self._execute_action(action)
                n_steps_done += 1
            # time.sleep(1)
            first = False

            iter_total = time.time() - iter_start
            print(f"Step {n_steps_done}: {iter_total*1000:.1f} ms")
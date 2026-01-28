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
            obs_dict = self._get_observation()
            t_inference_start = time.time()
            actions = self.policy_client.predict_action(obs_dict)
            print(f"inference_took {(time.time() - t_inference_start)*1000:.1f} ms")
            # print(f"{actions[0][0] - actions[7][0]}\t{actions[0][1] - actions[7][1]}\t{actions[0][2] - actions[7][2]}")
            # Execute first 8 actions
            for index, action in enumerate(actions[:8]):
                if n_steps_done >= n_steps:
                    break
                # print(action)
                self._execute_action(action)
                # pre_action = action.copy() if index == 0 else actions[index-1].copy()
                # action_to_execute = action
                # action_diff = np.linalg.norm(action - pre_action)
                # print(f"Action diff norm: {action_diff:.4f}")
                # max_step = 100  # Maximum allowed step size
                # if index != 0 and action_diff > max_step:
                #     # Calculate number of interpolation steps needed
                #     num_steps = int(np.ceil(action_diff / max_step))
                    
                #     # Generate interpolated actions
                #     for step in range(num_steps):
                #         t = (step + 1) / num_steps  # interpolation factor from (1/num_steps) to 1.0
                #         action_to_execute = pre_action + t * (action - pre_action)
                #         self._execute_action(action_to_execute)
                # else:
                #     # Normal execution for small movements or first action
                #     self._execute_action(action)
                # if first:
                #     time.sleep(0.05)
                n_steps_done += 1
            first = False

            iter_total = time.time() - iter_start
            print(f"Step {n_steps_done}: {iter_total*1000:.1f} ms")
from diff_eval_utils.controllers.base_controller import RobotController
import time


class ControlNetController(RobotController):

    def get_intervention_label(self):
        inhand_rgb = self._get_inhand_rgb()
        predicted_label, confidence, _ = self.policy_client.predict_intervention(inhand_rgb)
        return predicted_label, confidence

    def run(self, n_steps: int):
        """Execute simple sequential control.

        Each iteration:
        1. Get observation
        2. Query policy for actions
        3. Execute all actions sequentially
        """
        n_steps_done = 0
        intv_state = 0 # 0: no intv, 1: intv
        print("Starting simple sequential control...")
        while n_steps_done < n_steps:
            iter_start = time.time()

            # Get observation and action
            obs_dict = self._get_observation()
            actions = self.policy_client.predict_action(obs_dict)
            print(f"{actions[0][0] - actions[7][0]}\t{actions[0][1] - actions[7][1]}\t{actions[0][2] - actions[7][2]}")
            # Execute first 8 actions
            for action in actions[:8]:
                if n_steps_done >= n_steps:
                    break
                self._execute_action(action.copy())
                intv_label, confidence = self.get_intervention_label()
                if intv_label == 1:
                    if confidence < 0.85:
                        intv_label = 0

                if intv_label != intv_state:
                    intv_state = intv_label
                    print(f"Intervention state changed to {intv_state} with confidence {confidence:.3f}, re-inference...")
                    break
                intv_state = intv_label
                    

                n_steps_done += 1

            iter_total = time.time() - iter_start
            print(f"Step {n_steps_done}: {iter_total*1000:.1f} ms")

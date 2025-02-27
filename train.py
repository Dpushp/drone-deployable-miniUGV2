import mujoco
import mujoco.viewer
import numpy as np
import time

class TetherPulleySim:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None #mujoco.viewer.launch_passive(self.model, self.data, offscreen=True)
        self.action_dim = 5  # Number of actuators defined in XML

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0  # Reset control inputs
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action):
        self.data.ctrl[:] = action  # Apply action
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # Ensure state is updated
        
        state = self.get_state()
        reward = self.compute_reward(state, action)
        done = False  # Define termination criteria if needed
        
        return state, reward, done

    def get_state(self):
        pulley_pos = self.data.qpos[:3].copy()  # Pulley x, y, z position
        weight_pos = self.data.qpos[3:6].copy()  # Weight x, y, z position
        adhesion_value = self.data.ctrl[4]  # Adhesion control value
        test_box_pos = self.data.qpos[6:9].copy()  # Test box x, y, z (if movable)
        
        return np.concatenate([pulley_pos, weight_pos, [adhesion_value], test_box_pos])

    def compute_reward(self, state, action):
        # Placeholder for custom reward function
        reward = 0
        return reward

    def run(self, steps=500):
        self.reset()
        for step in range(steps):
            action = np.random.uniform(-1, 1, size=self.action_dim)  # Random actions
            state, reward, done = self.step(action)
            
            print(f"Step {step}:")
            print(f"Pulley Position: {state[:3]}")
            print(f"Weight Position: {state[3:6]}")
            print(f"Adhesion Value: {state[6]}")
            print(f"Test Box Position: {state[7:10]}")
            print("-----------------------------------")
            
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(0.02)  # Simulate real-time
            if done:
                break
        if self.viewer is not None:
            self.viewer.close()

if __name__ == "__main__":
    sim = TetherPulleySim("tether_pulley.xml")  # Update with correct XML path
    sim.run()

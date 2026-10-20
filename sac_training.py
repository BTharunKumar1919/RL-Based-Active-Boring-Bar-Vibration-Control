"""Train SAC policies for either a fixed or randomized plant/disturbance setup."""
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from config import PLANT, TRAINING
from env import VibrationEnv

class TrainingProgressCallback(BaseCallback):
    def __init__(self, on_progress=None, stop_event=None, verbose=0):
        super().__init__(verbose)
        self.on_progress=on_progress
        self.stop_event=stop_event
    def _on_step(self):
        if self.on_progress:
            total=max(int(self.model._total_timesteps),1)
            self.on_progress(min(1.0,self.num_timesteps/total), self.num_timesteps, total,
                             f"Training step {self.num_timesteps:,}/{total:,}")
        return not (self.stop_event and self.stop_event.is_set())

def train_sac(total_timesteps=None, seed=None, model_path="results/sac_vibration_model_7mode",
              plant_config=PLANT, generalized=False, plant_variation_pct=10.0,
              force_model_config=None, noise_config=None, on_progress=None, stop_event=None):
    if total_timesteps is None: total_timesteps=TRAINING.total_timesteps
    if seed is None: seed=TRAINING.seed
    if int(total_timesteps)<=0: raise ValueError("total_timesteps must be positive")
    set_random_seed(int(seed))
    env=Monitor(VibrationEnv(plant_config=plant_config, training_config=TRAINING,
        randomize_episode=True, generalized_plant=generalized,
        plant_variation_pct=plant_variation_pct, force_model_config=force_model_config,
        noise_config=noise_config))
    model=SAC("MlpPolicy", env, learning_rate=TRAINING.learning_rate,
        buffer_size=TRAINING.buffer_size, learning_starts=min(TRAINING.learning_starts, max(100,int(total_timesteps)//10)),
        batch_size=TRAINING.batch_size, tau=TRAINING.tau, gamma=TRAINING.gamma,
        train_freq=TRAINING.train_freq, gradient_steps=TRAINING.gradient_steps,
        seed=int(seed), verbose=0, device="auto", tensorboard_log="results/tensorboard")
    output=Path(model_path); output.parent.mkdir(parents=True,exist_ok=True)
    cb=TrainingProgressCallback(on_progress=on_progress,stop_event=stop_event)
    try:
        model.learn(total_timesteps=int(total_timesteps), callback=cb, progress_bar=False)
        # Save even if stopped: this is a partial checkpoint and metadata records it.
        model.save(str(output))
    finally:
        env.close()
    if on_progress: on_progress(1.0 if not (stop_event and stop_event.is_set()) else min(1.0, model.num_timesteps/max(int(total_timesteps),1)), model.num_timesteps, int(total_timesteps), "Checkpoint saved")
    return model

if __name__ == "__main__": train_sac()

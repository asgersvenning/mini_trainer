import torch
from torch.optim.swa_utils import AveragedModel

from mini_trainer.utils.loss import kl_distill_ema


def ema_lambda_per_update(half_life_steps : int | float, update_interval : int) -> float: # noqa: D103
    lam_step = (torch.tensor(1 / 2).log() / half_life_steps).exp().item()
    return lam_step ** update_interval


class EMATeacher(AveragedModel): # noqa: D101 TODO
    def __init__( # noqa: D107
            self, 
            enable : bool,
            total_steps : int,
            distill_start : int=0,
            update_rate : int=1,
            temperature : float=1.0, 
            *args, 
            **kwargs
        ):
        self._enabled, self.total_steps, self.distill_start, self.update_rate, self.temperature = \
            enable, total_steps, distill_start, update_rate, temperature
        if self._enabled:
            super().__init__(*args, **kwargs)
    
    def __bool__(self):
        return self._enabled
    
    def update_parameters(self, step, model):
        if not self._enabled or (step + 1) % self.update_rate != 0:
            return
        return super().update_parameters(model)

    def distill_rate(self, step: int) -> float:
        return float(min(1.0, max(0.0, (step - self.distill_start) / max(1.0, self.total_steps - self.distill_start))))
        
    def teach(
            self,
            step : int,
            input : torch.Tensor,
            student : torch.Tensor | list[torch.Tensor]
        ):
        """TODO.

        Returns:
          KL-divergence between the model EMA and the ``student`` logits.
            If the ``enabled=False``, ``0.0``.
        """
        dr = self.distill_rate(step)
        if isinstance(student, (list, tuple)):
            st_device, st_dtype = student[0].device, student[0].dtype
        else:
            st_device, st_dtype = student.device, student.dtype
        if not self._enabled or dr == 0.0:
            return torch.tensor(0.0, dtype=st_dtype, device=st_device)
        with torch.no_grad():
            is_train = self.training
            self.eval()
            ema_logits = self(input)
            self.train(is_train)
        return kl_distill_ema(student, ema_logits, T=self.temperature) * dr
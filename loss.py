import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class Loss(nn.Module):
    """The loss function.

    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.

    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).

    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.

    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
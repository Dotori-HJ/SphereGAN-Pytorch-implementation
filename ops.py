import torch
import torch.nn as nn

class HyperSphereLoss(nn.Module):
    def forward(self, input):
        '''
        Calcuate distance between input and N(North Pole) using hypersphere metrics.

        Woo Park, Sung, and Junseok Kwon.
        "Sphere Generative Adversarial Network Based on Geometric Moment Matching."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
        '''
        q = self.project_to_hypersphere(input)
        q_norm = torch.norm(q, dim=1) ** 2

        loss = (2 * q[:, -1]) / (1 + q_norm)
        return torch.mean(torch.acos(loss))

    def project_to_hypersphere(self, v):
        v_norm = torch.norm(v, dim=1, keepdim=True) ** 2
        a = 2 * v / (v_norm + 1)
        b = (v_norm - 1) / (v_norm + 1)
        return torch.cat([a, b], dim=1)

if __name__ == '__main__':
    # Test
    criterion = HyperSphereLoss()
    for i in range(1, 51):
        loss = criterion(torch.randn(64, 2) * i)
        print(loss)
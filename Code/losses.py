import torch.nn as nn

class ClusterLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		loss = 0
		y = y.squeeze(3).squeeze(2)
		for i in x:
			i = i.permute(1, 2, 0)
			i = i.reshape(-1, i.shape[2])
			length = i.shape[0]

			m = i / i.norm(dim=1, keepdim=True)
			n = (y / y.norm(dim=1, keepdim=True)).transpose(0, 1)

			z = m.mm(n)
			z = z.max(dim=1)[0]
			loss += (1. - z).sum() / float(length)
		return loss





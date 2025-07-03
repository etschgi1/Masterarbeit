import torch


class MessageNet(torch.nn.Module):
    """
    A flexible MLP to compute messages m_{u→v} from [c_u || c_v || e_{u→v}].

    Args:
      input_dim (int): dimensionality of the concatenated input vector
                       (typically 3 * hidden_dim).
      hidden_dim (int): dimensionality of the hidden layers (and output layer).
      num_layers (int): total number of Linear→GELU blocks. The final block
                        produces the message embedding of size hidden_dim.
                        Minimum = 1 (just one Linear→GELU from input_dim→hidden_dim).
      dropout (float):  dropout probability between layers (0.0 means no dropout). default=0.1
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        if type(dropout) is list:
            assert len(dropout) == num_layers - 1, "dropout list must have length num_layers - 1"
        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            if type(dropout) is list:
                layers.append(torch.nn.Dropout(dropout[i]))
            elif dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(torch.nn.GELU())


        self.net = torch.nn.Sequential(*layers)
    
    def __repr__(self):
        return super().__repr__()

    def forward(self, x):
        """
        x: Tensor of shape (E_total, input_dim), e.g. concatenated [c_u, c_v, e_uv]
        returns: Tensor of shape (E_total, hidden_dim), the message embeddings.
        """
        return self.net(x)

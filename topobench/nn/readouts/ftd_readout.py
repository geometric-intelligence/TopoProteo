import torch.nn as nn
import torch
from topobench.nn.readouts import AbstractZeroCellReadOut
from torch_geometric.utils import scatter


class FTDReadOut(AbstractZeroCellReadOut):
    ACT_MAP = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }

    def __init__(
        self,
        num_nodes,
        which_layer,
        fc_dim=None,
        fc_dropout=None,
        fc_act=None,
        out_channels=None,
        use_feature_encoder=True,
        feature_encoder_dim=None, #64
        graph_encoder_dim=None, #256
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, **kwargs)
        self.hidden_dim = kwargs["hidden_dim"]
        self.feature_encoder_dim = feature_encoder_dim
        self.graph_encoder_dim = graph_encoder_dim
        self.which_layer = which_layer
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.fc_act = fc_act
        self.fc_input_dim = graph_encoder_dim + feature_encoder_dim * 3
        self.out_channels = out_channels  # 1
        self.use_feature_encoder = use_feature_encoder
        self.readout_layers = self.build_readout_layers()
        self.feature_encoder = self.build_feature_encoder()
        self.graph_encoder = self.build_graph_encoder()

    def build_graph_encoder(self):
        return nn.Linear(self.hidden_dim, self.graph_encoder_dim)
   
    def build_feature_encoder(self):
        if self.use_feature_encoder:
            return nn.Linear(1, self.feature_encoder_dim)
        else:
            return nn.Identity()

    def encode_features(self, data):
        encoded_features = []
        for feature in self.which_layer:
            if feature in ["sex", "mutation", "age"]:
                feature_value = data.get(feature)
                encoded_features.append(self.feature_encoder(feature_value))
        return torch.cat(encoded_features, dim=1)

    def build_readout_layers(self):
        layers = []
        fc_layer_input_dim = self.fc_input_dim
        for fc_dim in self.fc_dim:
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_layer_input_dim, fc_dim),
                    self.ACT_MAP[self.fc_act],
                    nn.AlphaDropout(p=self.fc_dropout, inplace=True),
                )
            )
            fc_layer_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        return nn.Sequential(*layers)

    def concatenate_features(self, graph_features, demographic_features):
        batch_size = demographic_features.shape[0]
        graph_features = graph_features.view(batch_size, -1)
        return torch.cat([graph_features, demographic_features], dim=1)

    def forward(self, model_out, batch):
        encoded_graph = self.graph_encoder(model_out["x_0"])
        demographic_features = self.encode_features(batch)
        total_features = self.concatenate_features(
            encoded_graph, demographic_features
        )
        model_out["x_0"] = self.readout_layers(total_features)
        return model_out

    def __call__(self, model_out, batch) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)
        model_out["logits"] = model_out["x_0"]

        return model_out

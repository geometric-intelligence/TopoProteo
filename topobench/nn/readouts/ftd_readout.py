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
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, **kwargs)
        self.hidden_dim = kwargs["hidden_dim"]
        self.feature_output_dim = (
            self.hidden_dim
        )  # how big do you want sex, mutation, age to be encoded to
        self.which_layer = which_layer
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.fc_act = fc_act
        self.fc_input_dim = (
            self.hidden_dim * 3 * 2
        )  # how big are graph features + sex, mutation, age
        self.out_channels = out_channels  # 1
        self.use_feature_encoder = use_feature_encoder
        self.readout_layers = self.build_readout_layers()
        self.feature_encoder = self.build_feature_encoder()

    def scatter_node_features(self, x, batch):
        """Scatter node features to graph level.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        batch : torch_geometric.data.Batch
            Batch object containing the batched data.

        Returns
        -------
        torch.Tensor
            Graph-level features.
        """
        scatter_min = scatter(x, batch, dim=0, reduce="min")
        scatter_mean = scatter(x, batch, dim=0, reduce="mean")
        scatter_max = scatter(x, batch, dim=0, reduce="max")
        concat_scatter = torch.cat(
            [scatter_min, scatter_mean, scatter_max], dim=1
        )
        return concat_scatter

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

    def build_feature_encoder(self):
        if self.use_feature_encoder:
            return nn.Linear(1, self.feature_output_dim)
        else:
            return nn.Identity()

    def encode_features(self, data):
        sex = data.sex
        mutation = data.mutation
        age = data.age
        encoded_features = []
        for feature in self.which_layer:
            if feature in ["sex", "mutation", "age"]:
                feature_value = locals().get(feature)
                encoded_features.append(self.feature_encoder(feature_value))
        return torch.cat(encoded_features, dim=1)

    def concatenate_features(self, graph_features, demographic_features):
        batch_size = demographic_features.shape[0]
        graph_features = graph_features.view(batch_size, -1)
        return torch.cat([graph_features, demographic_features], dim=1)

    def forward(self, model_out, batch):
        scaterred_features = self.scatter_node_features(
            model_out["x_0"], batch["batch_0"]
        )
        demographic_features = self.encode_features(batch)
        total_features = self.concatenate_features(
            scaterred_features, demographic_features
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

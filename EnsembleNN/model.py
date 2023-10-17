#-*- coding:utf-8 -*-

import torch





class ResBlockModule(torch.nn.Module):
    """
    EnsembleNN Model Residual Block Module Class Setting

    in_size    : In Layer Features Size (Default = 1024)
    out_size   : Out Layer Features Size (Default = 1024)
    drop_rate  : Layer Dropout Rate (Default = 0.2)
    """
    def __init__(self, in_size: int = 1024, out_size: int = 1024, drop_rate: float = 0.2) -> None:
        super(ResBlockModule, self).__init__()

        self.BN = torch.nn.BatchNorm1d(num_features=in_size)
        self.Dense = torch.nn.Linear(in_features=in_size, out_features=out_size)
        self.Dropout = torch.nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x      : torch.Tensor (Default Shape = (Batch, in_size))
        Output
            return : torch.Tensor (Default Shape = (Batch, out_size))
        """
        return self.Dropout( self.Dense( self.BN(x) ) )


class EnsembleNN(torch.nn.Module):
    """
    EnsembleNN Model Class Setting

    in_size    : In Layer Features Size (int)
    out_size   : Out Layer Features Size (int)
    layer_size : Residual Dense Block Layer Features Size (Default = 1024)
    block_len  : Residual Dense Block Layer Length (Default = 6)
    drop_rate  : Residual Dense Block Layer Dropout Rate (Default = 0.2)
    """
    def __init__(self, in_size: int, out_size: int, layer_size: int = 1024, block_len: int = 6, drop_rate: float = 0.2) -> None:
        super(EnsembleNN, self).__init__()

        self.InLayer = torch.nn.Linear(in_features=in_size, out_features=layer_size)
        self.ResBlcokList = torch.nn.ModuleList(
            [ResBlockModule(in_size=layer_size, out_size=layer_size, drop_rate=drop_rate) for _ in range(block_len)]
        )
        self.OutLayer = torch.nn.Linear(in_features=layer_size, out_features=out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x      : torch.Tensor (Default Shape = (Batch, in_size))
        Output
            return : torch.Tensor (Default Shape = (Batch, out_size))
            * Caution:  Used Log_Softmax,  To use Overlay_Func, Add Softmax after Training it.
        """
        x = torch.nn.functional.relu( self.InLayer( x.float() ) )
        for ResBlcok in self.ResBlcokList:
            x = torch.nn.functional.relu( x + ResBlcok(x) )
        x = self.OutLayer(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class OverlayFunc(object):
    """
    This OverlayFunc Class is Projects the Individual Outputs of the EnsembleNN Model based on the cat_info.
    For Execution, Check the run Function of the Class.

    Input
        cat_info = Number of Categories for each Sub Model (list[sub_model_1_cat_len, sub_model_2_cat_len, ..., sub_model_N_cat_len])
    """
    def __init__(self, cat_info: list) -> None:
        self.model_len = len(cat_info)
        self.cat_len = sum(cat_info)
        self.proj_range = self.get_projection_range(cat_info)

    def prefix_sum(self, arr: list) -> list:
        prefixSum = [arr[0]]
        for i in range(1, len(arr)):
            prefixSum.append(prefixSum[i-1] + arr[i])
        return prefixSum

    def get_projection_range(self, arr: list) -> list:
        prf_arr = [0] + self.prefix_sum(arr)
        return [(x, y, i) for i, (x, y) in enumerate(zip(prf_arr[:-1], prf_arr[1:]))]

    def run(self, data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Input
            data   : Combining the output of Sub Models with the Input of the Combined EnsembleNN Model (Batch, Num of Categories)
            output : As the Output of the EnsembleNN Model, the Probability of Selecting a Sub Model for the Result (Batch, Num of Sub Models)
        Output
            return : Projected Final EnsembleNN Result (Batch, Num of Categories)
        """
        if self.cat_len != data.shape[1]:
            raise Exception("DataShapeError: The shape of categories_len({}) and the shape of data({}) are different.".format(
                self.cat_len, data.shape[1]))

        if self.model_len != output.shape[1]:
            raise Exception("DataShapeError: The shape of model_len({}) and the shape of output({}) are different.".format(
                self.model_len, output.shape[1]))

        return torch.cat([data[:, i:j] * output[:, k].unsqueeze(dim = 1) for i, j, k in self.proj_range], dim = 1)
    


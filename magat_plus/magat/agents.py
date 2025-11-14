from typing import Sequence
import math

import torch
import torch.nn.functional as F

from magat_plus.generate_additional_data import any_additional_data
from magat_plus.magat.model.model_selection import get_gnn_module


class CNN(torch.nn.Module):
    convs: Sequence[torch.nn.Conv2d]
    batch_norms: Sequence[torch.nn.BatchNorm2d]
    compressMLP: Sequence[torch.nn.Linear]

    def __init__(
        self,
        *,
        numChannel,
        numStride,
        convW,
        convH,
        nMaxPoolFilterTaps,
        numMaxPoolStride,
        embedding_sizes,
    ):
        super().__init__()

        convs = []
        batch_norms = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convs.append(
                torch.nn.Conv2d(
                    in_channels=numChannel[l],
                    out_channels=numChannel[l + 1],
                    kernel_size=nFilterTaps[l],
                    stride=numStride[l],
                    padding=nPaddingSzie[l],
                    bias=True,
                )
            )
            batch_norms.append(torch.nn.BatchNorm2d(num_features=numChannel[l + 1]))
            # convl.append(torch.nn.ReLU(inplace=True))

            convW = (
                int((convW - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            convH = (
                int((convH - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                convW = int((convW - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                convH = int((convH - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/

        self.convs = torch.nn.ModuleList(convs)
        self.batch_norms = torch.nn.ModuleList(batch_norms)

        numFeatureMap = numChannel[-1] * convW * convH

        numCompressFeatures = [numFeatureMap] + embedding_sizes

        self.nMaxPoolFilterTaps = nMaxPoolFilterTaps
        self.numMaxPoolStride = numMaxPoolStride

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x)
            x = batch_norm(x)
            x = F.relu(x)

            if i % 2 == 0:
                x = F.max_pool2d(
                    x, kernel_size=self.nMaxPoolFilterTaps, stride=self.numMaxPoolStride
                )
        x = x.reshape((x.shape[0], -1))
        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=(1, 1),
        residual=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        layers,
        in_channels=3,
        num_classes=128,
        channels=(32, 64, 128, 128),
        out_map=False,
        out_middle=False,
        pool_size=2,
        arch="D",
    ):
        super(ResNet, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        self.conv1 = torch.nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(channels[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=1)

        if num_classes > 0:
            self.avgpool = torch.nn.AvgPool2d(pool_size)
            self.fc = torch.nn.Conv2d(
                self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True
            )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True
    ):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=(
                    (1, 1)
                    if dilation == 1
                    else (dilation // 2 if new_level else dilation, dilation)
                ),
                residual=residual,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    residual=residual,
                    dilation=(dilation, dilation),
                )
            )

        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for l in self.layer1:
            l.reset_parameters()
        for l in self.layer2:
            l.reset_parameters()
        for l in self.layer3:
            l.reset_parameters()
        self.fc.reset_parameters()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        y.append(x)

        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        # print("x - postCNN:", x.shape)
        if self.out_map:
            x = self.fc(x)
        else:
            x_TMP = self.avgpool(x)
            # print('x - after avgpool', x_TMP.shape)
            x = self.fc(x_TMP)
            # print('x - after FCN', x.shape)
            # print('check x == x_tmp', x-x_TMP)
            # x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class ResNetLarge_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet([1, 1, 1], in_channels=in_channels, out_map=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(
            in_features=1152, out_features=output_size, bias=True
        )

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class ResNetSmaller_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet(
            [1, 1, 1],
            in_channels=in_channels,
            out_map=False,
            channels=(16, 32, 64, 64),
            num_classes=64,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(in_features=576, out_features=output_size, bias=True)

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class ResNetVSmaller_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet(
            [1, 1, 1],
            in_channels=in_channels,
            out_map=False,
            channels=(8, 16, 32, 32),
            num_classes=32,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(in_features=288, out_features=output_size, bias=True)

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class AdditionalDataProcessor(torch.nn.Module):
    def __init__(
        self,
        additional_data_idx,
        output_size,
        lin_x=False,
    ):
        super().__init__()
        self.greedy_action_mlp = None
        self.prev_actions_mlp = None
        self.lin_x = None
        if additional_data_idx[1] is not None:
            greedy_action_mlp_lins = []
            greedy_action_mlp_sizes = [5, output_size]
            for i in range(len(greedy_action_mlp_sizes) - 1):
                greedy_action_mlp_lins.append(
                    torch.nn.Linear(
                        in_features=greedy_action_mlp_sizes[i],
                        out_features=greedy_action_mlp_sizes[i + 1],
                    )
                )
            self.greedy_action_mlp = torch.nn.ModuleList(greedy_action_mlp_lins)
        if additional_data_idx[2] is not None:
            _, num_prev_actions = additional_data_idx[2]
            prev_actions_mlp_lins = []
            prev_actions_mlp_sizes = [num_prev_actions * 5, output_size, output_size]
            for i in range(len(prev_actions_mlp_sizes) - 1):
                prev_actions_mlp_lins.append(
                    torch.nn.Linear(
                        in_features=prev_actions_mlp_sizes[i],
                        out_features=prev_actions_mlp_sizes[i + 1],
                    )
                )
            self.prev_actions_mlp = torch.nn.ModuleList(prev_actions_mlp_lins)
        if lin_x:
            self.lin_x = torch.nn.Linear(
                in_features=output_size, out_features=output_size
            )

    def reset_parameters(self):
        if self.greedy_action_mlp is not None:
            for lin in self.greedy_action_mlp:
                lin.reset_parameters()
        if self.prev_actions_mlp is not None:
            for lin in self.prev_actions_mlp:
                lin.reset_parameters()
        if self.lin_x is not None:
            self.lin_x.reset_parameters()

    def forward(self, x, data):
        if self.lin_x is not None:
            x = self.lin_x(x)
            x = F.relu(x)
        if self.greedy_action_mlp is not None:
            greedy_action = data.greedy_action
            for lin in self.greedy_action_mlp:
                greedy_action = lin(greedy_action)
                greedy_action = F.relu(greedy_action)
            x = x + greedy_action
        if self.prev_actions_mlp is not None:
            prev_actions = data.prev_actions
            for lin in self.prev_actions_mlp:
                prev_actions = lin(prev_actions)
                prev_actions = F.relu(prev_actions)
            x = x + prev_actions
        return x


class ResetablleSequential(torch.nn.Sequential):
    def reset_parameters(self):
        for module in self:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x


class ParameterShareCNN(torch.nn.Module):
    def __init__(self, cnn, mode, num_channels=None):
        super().__init__()
        self.cnn = cnn
        self.mode = mode
        self.num_channels = num_channels

    def reset_parameters(self):
        pass

    def forward(self, x):
        if self.mode == "pad-with-zeroes":
            padding_shape = (x.shape[0], self.num_channels - 1, *x.shape[2:])
            padding = torch.zeros(
                padding_shape, device=x.device, dtype=x.dtype, layout=x.layout
            )
            x = torch.cat([padding[:, 0:1], x, padding[:, 1:]], dim=1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}.")
        return self.cnn(x)


def get_cnn(cnn_mode, cnn_num_input_channels, cnn_output_size, inW, inH):
    if cnn_mode == "basic-CNN":
        cnn = CNN(
            numChannel=[cnn_num_input_channels, 32, 32, 64, 64, 128],
            numStride=[1, 1, 1, 1, 1],
            convW=inW,
            convH=inH,
            nMaxPoolFilterTaps=2,
            numMaxPoolStride=2,
            embedding_sizes=[cnn_output_size],
        )
    elif cnn_mode == "ResNetLarge_withMLP":
        cnn = ResNetLarge_withMLP(
            output_size=cnn_output_size,
            embedding_sizes=[cnn_output_size],
            in_channels=cnn_num_input_channels,
        )
    elif cnn_mode == "MLP":
        channels = [cnn_num_input_channels, 32, 32, cnn_output_size]
        mlp = []
        mlp.append(
            torch.nn.Linear(
                in_features=channels[0], out_features=channels[1], bias=True
            )
        )
        for i in range(1, len(channels) - 1):
            mlp.append(torch.nn.ReLU())
            mlp.append(
                torch.nn.Linear(
                    in_features=channels[i],
                    out_features=channels[i + 1],
                    bias=True,
                )
            )
        cnn = ResetablleSequential(*mlp)
    elif cnn_mode == "flatten-MLP":
        channels = [inW * inH * cnn_num_input_channels, 8, cnn_output_size]
        mlp = []
        mlp.append(torch.nn.Flatten())
        mlp.append(
            torch.nn.Linear(
                in_features=channels[0], out_features=channels[1], bias=True
            )
        )
        for i in range(1, len(channels) - 1):
            mlp.append(torch.nn.ReLU())
            mlp.append(
                torch.nn.Linear(
                    in_features=channels[i],
                    out_features=channels[i + 1],
                    bias=True,
                )
            )
        cnn = ResetablleSequential(*mlp)
    else:
        raise ValueError(f"Unsupported cnn_mode: {cnn_mode}.")

    return cnn


class MAGATPlus(torch.nn.Module):
    def __init__(
        self,
        *,
        FOV,
        numInputFeatures,
        num_attention_heads,
        use_dropout,
        gnn_type,
        gnn_kwargs,
        concat_attention,
        num_classes=5,
        cnn_mode="basic-CNN",
        cnn_output_size=None,
        num_gnn_layers=None,
        embedding_sizes_gnn=None,
        use_edge_weights=False,
        use_edge_attr=False,
        edge_dim=None,
        model_residuals=None,
        module_residual=[],
        additional_data_idx=[None, None, None],
        lin_x_before_additional_data=False,
        use_edge_attr_for_messages=None,
        edge_attr_processor=None,
    ):
        super().__init__()

        assert concat_attention is True, "Currently only support concat attention."

        if embedding_sizes_gnn is None:
            assert num_gnn_layers is not None
            embedding_sizes_gnn = num_gnn_layers * [numInputFeatures]
        else:
            if num_gnn_layers is not None:
                assert num_gnn_layers == len(embedding_sizes_gnn)
            else:
                num_gnn_layers = len(embedding_sizes_gnn)

        inW = (FOV + 1) * 2 + 1
        inH = (FOV + 1) * 2 + 1

        if cnn_output_size is None:
            cnn_output_size = numInputFeatures
        cnn_num_input_channels = 3
        if additional_data_idx[0] is not None:
            cnn_num_input_channels = 4

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################

        self.cnn = get_cnn(
            cnn_mode=cnn_mode,
            cnn_num_input_channels=cnn_num_input_channels,
            cnn_output_size=cnn_output_size,
            inW=inW,
            inH=inH,
        )

        self.use_edge_attr_for_messages = use_edge_attr_for_messages
        self.edge_attr_cnn = None
        if use_edge_attr_for_messages is not None:
            if edge_attr_processor is None:
                edge_attr_processor = cnn_mode
            num_edge_attr_input_channels = 3

            self.edge_attr_cnn = get_cnn(
                cnn_mode=edge_attr_processor,
                cnn_num_input_channels=num_edge_attr_input_channels,
                cnn_output_size=cnn_output_size,
                inW=inW,
                inH=inH,
            )
            if edge_dim is None:
                edge_dim = cnn_output_size
            assert edge_dim == cnn_output_size

        self.additional_data_processor = AdditionalDataProcessor(
            additional_data_idx=additional_data_idx,
            output_size=cnn_output_size,
            lin_x=lin_x_before_additional_data,
        )
        self.numFeatures2Share = cnn_output_size

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################
        self.gnn = get_gnn_module(
            in_channels=self.numFeatures2Share,
            embedding_sizes=embedding_sizes_gnn,
            num_attention_heads=num_attention_heads,
            num_gnn_layers=num_gnn_layers,
            model_type=gnn_type,
            use_edge_weights=use_edge_weights,
            use_edge_attr=use_edge_attr,
            edge_dim=edge_dim,
            model_residuals=model_residuals,
            **gnn_kwargs,
        )

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        actions_mlp_sizes = [
            num_attention_heads * embedding_sizes_gnn[-1],
            embedding_sizes_gnn[-1],
            num_classes,
        ]

        actionsfc = []
        for i in range(len(actions_mlp_sizes) - 1):
            actionsfc.append(
                torch.nn.Linear(
                    in_features=actions_mlp_sizes[i],
                    out_features=actions_mlp_sizes[i + 1],
                )
            )
        self.use_dropout = use_dropout
        self.actionsMLP = torch.nn.ModuleList(actionsfc)

        self.cnn_to_out_lin = None
        for res in module_residual:
            if res == "cnn-to-out":
                self.cnn_to_out_lin = torch.nn.Linear(
                    cnn_output_size, actions_mlp_sizes[0]
                )
            else:
                raise ValueError(f"Unsupported module_residual: {res}.")

    def reset_parameters(self):
        self.cnn.reset_parameters()
        if self.edge_attr_cnn is not None:
            self.edge_attr_cnn.reset_parameters()
        self.additional_data_processor.reset_parameters()
        self.gnn.reset_parameters()
        for lin in self.actionsMLP:
            lin.reset_parameters()
        if self.cnn_to_out_lin is not None:
            self.cnn_to_out_lin.reset_parameters()

    def in_simulation(self, value):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, data):
        cnn_out = self.cnn(x)
        cnn_out = self.additional_data_processor(cnn_out, data)

        x = cnn_out

        gnn_input_kwargs = dict()
        edge_attr = None
        if self.edge_attr_cnn is not None:
            if edge_attr is not None and edge_attr.shape[0] > 0:
                edge_attr = self.edge_attr_cnn(edge_attr)
            elif data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                edge_attr = self.edge_attr_cnn(data.edge_attr)
            else:
                edge_attr = None
            gnn_input_kwargs["edge_attr"] = edge_attr
        x = self.gnn(x, data, **gnn_input_kwargs)

        if self.cnn_to_out_lin is not None:
            res_out = self.cnn_to_out_lin(cnn_out)
            res_out = F.relu(res_out)
            x = res_out + x

        for lin in self.actionsMLP[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.actionsMLP[-1](x)

        return x


def _decode_residual_args(args):
    if args.module_residual is None:
        return []

    return args.module_residual.split("+")


_GNN_DEF_KEYS = [
    "imitation_learning_model",
    "embedding_size",
    "num_gnn_layers",
    "num_attention_heads",
    "attention_mode",
    "edge_dim",
    "model_residuals",
    "use_edge_weights",
    "use_edge_attr",
    "use_edge_attr_for_messages",
    "edge_attr_processor",
]


def _decode_args(args: dict) -> dict:
    args = {key: args[key] for key in _GNN_DEF_KEYS}
    model_kwargs = {
        key: args[key]
        for key in [
            "num_gnn_layers",
            "use_edge_attr",
            "use_edge_weights",
            "edge_dim",
            "model_residuals",
            "use_edge_attr_for_messages",
            "edge_attr_processor",
        ]
    }
    gnn_kwargs = dict()
    if args["imitation_learning_model"] == "MAGATPlus":
        gnn_kwargs = gnn_kwargs | {
            "attentionMode": args["attention_mode"],
            "use_edge_attr_for_messages": args["use_edge_attr_for_messages"],
        }
        gnn_type = "MAGATPlus"
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args['imitation_learning_model']}."
        )
    model_kwargs = model_kwargs | {
        "gnn_kwargs": gnn_kwargs,
        "gnn_type": gnn_type,
    }
    return model_kwargs


def get_model(args, device) -> tuple[torch.nn.Module, bool, dict]:
    _, additional_data_idx = any_additional_data(args)

    common_kwargs = dict(
        FOV=args.obs_radius,
        numInputFeatures=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        cnn_mode=args.cnn_mode,
        use_dropout=True,
        concat_attention=True,
        num_classes=5,
        module_residual=_decode_residual_args(args),
        additional_data_idx=additional_data_idx,
    )
    model_kwargs = _decode_args(vars(args))
    dataset_kwargs = {
        "use_edge_attr": model_kwargs["use_edge_attr"],
        "use_edge_attr_for_messages": args.use_edge_attr_for_messages,
    }

    model = MAGATPlus(**common_kwargs, **model_kwargs).to(device)
    model.reset_parameters()

    return model, dataset_kwargs

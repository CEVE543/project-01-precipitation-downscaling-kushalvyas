import torch
import torch.nn as nn 

def get_activation(name):
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError

def conv_bn_relu_block(in_channels , out_channels, kernel_size=(3,3), activation='leaky_relu', use_bn=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)

def conv_and_down(in_channels , out_channels, kernel_size=(3,3), activation='leaky_relu', use_bn=True):
    layers = []
    layers.append(conv_bn_relu_block(in_channels , out_channels, kernel_size=kernel_size, activation=activation, use_bn=use_bn))
    layers.append(nn.AvgPool2d(2))
    layers.append(conv_bn_relu_block(out_channels , out_channels, kernel_size=kernel_size, activation=activation, use_bn=use_bn))
    return nn.Sequential(*layers)

def conv_and_up(in_channels , out_channels, kernel_size=(3,3), activation='leaky_relu', use_bn=True):
    layers = []
    layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
    layers.append(conv_bn_relu_block(in_channels , out_channels, kernel_size=kernel_size, activation=activation, use_bn=use_bn))
    layers.append(conv_bn_relu_block(out_channels , out_channels, kernel_size=kernel_size, activation=activation, use_bn=use_bn))
    return nn.Sequential(*layers)



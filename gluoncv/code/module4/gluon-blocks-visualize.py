import mxnet as mx 

from mxnet import nd, viz
from mxnet.gluon import nn, model_zoo

################################################################################

# get VGG11 (visual geometry group: a group from oxford university) network from model zoo
vgg11 = model_zoo.vision.vgg11(pretrained=True)

## simple visualizing by printing it
# VGG11 is composed of multiple layer: con2D, maxpool, dense, ...
# main blocks is features (with further children blocks) and output block
print(vgg11)
# VGG(
#   (features): HybridSequential(
#     (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): Activation(relu)
#     (2): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#     (3): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): Activation(relu)
#     (5): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#     (6): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): Activation(relu)
#     (8): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): Activation(relu)
#     (10): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#     (11): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): Activation(relu)
#     (13): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (14): Activation(relu)
#     (15): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#     (16): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): Activation(relu)
#     (18): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): Activation(relu)
#     (20): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#     (21): Dense(25088 -> 4096, Activation(relu))
#     (22): Dropout(p = 0.5, axes=())
#     (23): Dense(4096 -> 4096, Activation(relu))
#     (24): Dropout(p = 0.5, axes=())
#   )
#   (output): Dense(4096 -> 1000, linear)
# )

## summary function
# demo with dummy data (= ones)
# check Dense-23: it has over 100 million parameters (= 102764544)!
print( vgg11.summary(nd.ones(shape=(1,3,224,224))) )
# --------------------------------------------------------------------------------
#         Layer (type)                                Output Shape         Param #
# ================================================================================
#                Input                            (1, 3, 224, 224)               0
#             Conv2D-1                           (1, 64, 224, 224)            1792
#         Activation-2                           (1, 64, 224, 224)               0
#          MaxPool2D-3                           (1, 64, 112, 112)               0
#             Conv2D-4                          (1, 128, 112, 112)           73856
#         Activation-5                          (1, 128, 112, 112)               0
#          MaxPool2D-6                            (1, 128, 56, 56)               0
#             Conv2D-7                            (1, 256, 56, 56)          295168
#         Activation-8                            (1, 256, 56, 56)               0
#             Conv2D-9                            (1, 256, 56, 56)          590080
#        Activation-10                            (1, 256, 56, 56)               0
#         MaxPool2D-11                            (1, 256, 28, 28)               0
#            Conv2D-12                            (1, 512, 28, 28)         1180160
#        Activation-13                            (1, 512, 28, 28)               0
#            Conv2D-14                            (1, 512, 28, 28)         2359808
#        Activation-15                            (1, 512, 28, 28)               0
#         MaxPool2D-16                            (1, 512, 14, 14)               0
#            Conv2D-17                            (1, 512, 14, 14)         2359808
#        Activation-18                            (1, 512, 14, 14)               0
#            Conv2D-19                            (1, 512, 14, 14)         2359808
#        Activation-20                            (1, 512, 14, 14)               0
#         MaxPool2D-21                              (1, 512, 7, 7)               0
#        Activation-22                                   (1, 4096)               0
#             Dense-23                                   (1, 4096)       102764544
#           Dropout-24                                   (1, 4096)               0
#        Activation-25                                   (1, 4096)               0
#             Dense-26                                   (1, 4096)        16781312
#           Dropout-27                                   (1, 4096)               0
#             Dense-28                                   (1, 1000)         4097000
#               VGG-29                                   (1, 1000)               0
# ================================================================================
# Parameters in forward computation graph, duplicate included
#    Total params: 132863336
#    Trainable params: 132863336
#    Non-trainable params: 0
# Shared params in forward computation graph: 0
# Unique parameters in model: 132863336
# --------------------------------------------------------------------------------

## visualize computational graph use plot_network function
# you need graphviz for this!
# sudo apt install graphviz on ubuntu
# digraph = viz.plot_network(
#     vgg11(mx.sym.var('data')),
#     shape={'data':(1,3,224,224)},
#     node_attrs={"shape":"oval", "fixedsize":"false"}
# )
# digraph.view()

## xml tool: netron
# you need https://github.com/lutzroeder/netron to load the .json file
# vgg11.hybridize()
# vgg11(nd.ones((1,1,224,224)))
# vgg11.export('vgg11')

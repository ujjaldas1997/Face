import mxnet as mx
net = mx.sym.load('hr101-symbol.json')
#mx.viz.plot_network(net)
model = mx.mod.Module(symbol=net, data_names=['data'], label_names=None)
model.bind(data_shapes=[('data', (1, 3, 224, 224))])
model.init_params()
model.save_checkpoint('my', 0)

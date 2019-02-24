import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import time
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
	print(layer_index)
	_, conv = list(model.features._modules.items())[layer_index]
	bn_layer_index = layer_index +1
	relu_layer_index =layer_index+2
	bn = list(model.features._modules.items())[bn_layer_index]
	relu = list(model.features._modules.items())[relu_layer_index]
	next_conv = None
	next_bn= None
	offset = 1
	boffset =1
	while layer_index + boffset <  len(list(model.features._modules.items())):
		res =  list(model.features._modules.items())[layer_index+boffset]
		if isinstance(res[1], torch.nn.modules.BatchNorm2d):
			next_bname, next_bn = res
			break
		bboffset = boffset + 1
	while layer_index + offset <  len(list(model.features._modules.items())):
		res =  list(model.features._modules.items())[layer_index+offset]
		if isinstance(res[1], torch.nn.modules.conv.Conv2d):
			next_name, next_conv = res
			break
		offset = offset + 1
	# batchnorm
	new_bn =torch.nn.BatchNorm2d(num_features =conv.out_channels - 1, eps=1e-05, momentum=0.1, affine=True)
	old_bweights = bn.weight.data.cpu().numpy()
	new_bweights = new_bn.weight.data.cpu().numpy()

	new_bweights[:filter_index] = old_bweights[:filter_index]
	new_bweights[filter_index :] = old_bweights[filter_index + 1 :]
	
	if torch.cuda.is_available():
		new_bn.weight.data = torch.from_numpy(new_bweights).cuda()
	else:
		new_bn.weight.data = torch.from_numpy(new_bweights)

	bn_bias_numpy = bn.bias.data.cpu().numpy()

	bias = np.zeros(shape = (bn_bias_numpy.shape[0] - 1), dtype = np.float32)
	bias[:filter_index] = bn_bias_numpy[:filter_index]
	bias[filter_index : ] = bn_bias_numpy[filter_index + 1 :]
	
	if torch.cuda.is_available():
		new_bn.bias.data = torch.from_numpy(bias).cuda()
	else:
		new_bn.bias.data = torch.from_numpy(bias)	

	runavg_numpy = bn.running_mean.data.cpu().numpy()

	running_mean = np.zeros(shape = (runavg_numpy.shape[0] - 1), dtype = np.float32)
	running_mean[:filter_index] = runavg_numpy[:filter_index]
	running_mean[filter_index : ] = runavg_numpy[filter_index + 1 :]
	
	if torch.cuda.is_available():
		new_bn.running_mean.data = torch.from_numpy(running_mean).cuda()
	else:
		new_bn.running_mean.data = torch.from_numpy(running_mean)
	
	runvar_numpy = bn.running_var.data.cpu().numpy()

	running_var = np.zeros(shape = (runvar_numpy.shape[0] - 1), dtype = np.float32)
	running_var[:filter_index] = runvar_numpy[:filter_index]
	running_var[filter_index : ] = runvar_numpy[filter_index + 1 :]
	
	if torch.cuda.is_available():
		new_bn.running_var.data = torch.from_numpy(running_var).cuda()
	else:
		new_bn.running_var.data = torch.from_numpy(running_var)

	new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - 1,
			kernel_size = conv.kernel_size, \
			stride = conv.stride,
			padding = conv.padding,
			dilation = conv.dilation,
			groups = conv.groups,
			bias = True if conv.bias is not None else False)

	old_weights = conv.weight.data.cpu().numpy()
	new_weights = new_conv.weight.data.cpu().numpy()

	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	
	if torch.cuda.is_available():
		new_conv.weight.data = torch.from_numpy(new_weights).cuda()
	else:
		new_conv.weight.data = torch.from_numpy(new_weights)

	bias_numpy = conv.bias.data.cpu().numpy()

	bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	bias[:filter_index] = bias_numpy[:filter_index]
	bias[filter_index : ] = bias_numpy[filter_index + 1 :]
	
	if torch.cuda.is_available():
		new_conv.bias.data = torch.from_numpy(bias).cuda()
	else:
		new_conv.bias.data = torch.from_numpy(bias)
	
	if not next_bn is None:
		next_new_bn = torch.nn.BatchNorm2d(num_features =conv.out_channels - 1, eps=1e-05, momentum=0.1, affine=True)

		old_weights = next_bn.weight.data.cpu().numpy()
		new_weights = next_new_bn.weight.data.cpu().numpy()

		new_weights[: filter_index] = old_weights[ : filter_index]
		new_weights[filter_index :] = old_weights[filter_index + 1 :]
		
		if torch.cuda.is_available():
			next_new_bn.weight.data = torch.from_numpy(new_weights).cuda()
		else:
			next_new_bn.weight.data = torch.from_numpy(new_weights)

		next_new_bn.bias.data = next_bn.bias.data
		next_new_bn.running_mean.data = next_bn.running_mean.data
		next_new_bn.running_mean.data = next_bn.running_mean.data

	if not next_conv is None:
		next_new_conv = \
			torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
				out_channels =  next_conv.out_channels, \
				kernel_size = next_conv.kernel_size, \
				stride = next_conv.stride,
				padding = next_conv.padding,
				dilation = next_conv.dilation,
				groups = next_conv.groups,
				bias = True if next_conv.bias is not None else False)

		old_weights = next_conv.weight.data.cpu().numpy()
		new_weights = next_new_conv.weight.data.cpu().numpy()

		new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
		new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
		
		if torch.cuda.is_available():
			next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
		else:
			next_new_conv.weight.data = torch.from_numpy(new_weights)

		next_new_conv.bias.data = next_conv.bias.data
		# next_new_conv.running_mean.data = next_conv.running_mean.data
		# next_new_conv.running_var.data = next_conv.running_var.data

	if not next_conv is None:
	 	features = torch.nn.Sequential(
	            *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
	            	[new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
	 	del model.features
	 	del conv

	 	model.features = features

	else:
		#Prunning the last conv layer. This affects the first linear layer of the classifier.
		model.features = torch.nn.Sequential(
				*(replace_layers(model.features, i, [layer_index], \
					[new_conv]) for i, _ in enumerate(model.features)))
		layer_index = 0
		old_linear_layer = None
		for _, module in model.classifier._modules.items():
			if isinstance(module, torch.nn.Linear):
				old_linear_layer = module
				break
			layer_index = layer_index  + 1

		if old_linear_layer is None:
			raise BaseException("No linear laye found in classifier")
		params_per_input_channel = old_linear_layer.in_features/conv.out_channels
		print(old_linear_layer.out_features)
		new_linear_layer = \
			torch.nn.Linear(int(old_linear_layer.in_features - params_per_input_channel), 
				int(old_linear_layer.out_features))

		old_weights = old_linear_layer.weight.data.cpu().numpy()
		new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

		new_weights[:, : int(filter_index * params_per_input_channel)] = \
			old_weights[:, : int(filter_index * params_per_input_channel)]
		new_weights[:, int(filter_index * params_per_input_channel ):] = \
			old_weights[:, int((filter_index + 1) * params_per_input_channel) :]

		new_linear_layer.bias.data = old_linear_layer.bias.data

		
		if torch.cuda.is_available():
			new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()
		else:
			new_linear_layer.weight.data = torch.from_numpy(new_weights)

		classifier = torch.nn.Sequential(
			*(replace_layers(model.classifier, i, [layer_index], \
				[new_linear_layer]) for i, _ in enumerate(model.classifier)))

		del model.classifier
		del next_conv
		del conv
		model.classifier = classifier

	return model

if __name__ == '__main__':
	# model = models.vgg16(pretrained=True)
	model = torch.load('vgg11cifar.net')
	model.train()

	t0 = time.time()
	model = prune_vgg16_conv_layer(model, 1, 10)
	print ("The prunning took", time.time() - t0)
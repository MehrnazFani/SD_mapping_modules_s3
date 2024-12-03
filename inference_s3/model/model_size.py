
import torch.nn as nn

class model_size(nn.Module):
    expansion = 1

    def __init__(self, input_dimension, layer_index=1):
        self.input_dimension =  input_dimension
        self.layer_index = layer_index

    def main(self, model):
        text = ""
        output_size = []
        model_outputs = []
        for layer in model:
            permission_to_print = False
            if type(layer) == nn.Conv2d:
                output_size, \
                text = self.calc_out_conv_layers(self.input_dimension[0], 
                                                 self.input_dimension[1], 
                                                 layer.kernel_size, 
                                                 layer.padding, 
                                                 layer.dilation, 
                                                 layer.stride,
                                                 layer.in_channels,
                                                 layer.out_channels) 

                self.input_dimension[0], self.input_dimension[1], self.input_dimension[2] = output_size

                model_outputs.append(output_size)
                
                permission_to_print = True

            elif type(layer) == nn.ConvTranspose2d:
                output_size, \
                text = self.calc_out_deconv_layers(self.input_dimension[0], 
                                                   self.input_dimension[1], 
                                                   layer.kernel_size, 
                                                   layer.padding, 
                                                   layer.dilation, 
                                                   layer.stride,
                                                   layer.in_channels,
                                                   layer.out_channels) 
                
                self.input_dimension[0], self.input_dimension[1], self.input_dimension[2] = output_size

                model_outputs.append(output_size)

                permission_to_print = True
                
            elif type(layer) == nn.MaxPool2d:
                output_size, \
                text = self.calc_maxpooling_layers(self.input_dimension[0], 
                                                   self.input_dimension[1], 
                                                   layer.stride) 
        
                self.input_dimension[0], self.input_dimension[1] = output_size

                model_outputs.append(output_size)

                permission_to_print = True

            elif type(layer) == nn.Upsample:
                output_size, \
                text = self.calc_upsampling_layers(self.input_dimension[0], 
                                                   self.input_dimension[1],
                                                   layer.scale_factor) 
                
                self.input_dimension[0], self.input_dimension[1] = output_size

                model_outputs.append(output_size)

                permission_to_print = True
 
            if  permission_to_print:
                print(text)

            if type(layer) != nn.ReLU:
                self.layer_index += 1

        return output_size, self.layer_index, model_outputs

    def calc_out_conv_layers(self, in_h, in_w, kernel_size, padding, dilation, stride, in_channels, out_channels):
        # for ker, pad, dil, stri in zip(kernel_size, padding, dilation, stride): # Just for a single layer
        out_h = int(((in_h + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1)/stride[0]) + 1)
        out_w = int(((in_w + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1)/stride[1]) + 1)

        output_size = [out_h, out_w, out_channels]
        text = f"layer_index: {self.layer_index} - type: Conv. - input-size: {[in_h, in_w, in_channels]} - Output-size: {output_size}."

        return output_size, text

    def calc_out_deconv_layers(self, in_h, in_w, kernel_size, padding, dilation, stride, in_channels, out_channels):
        # for ker, pad, dil, stri in zip(kernel_size, padding, dilation, stride): # Just for a single layer
        out_h = int((in_h-1)*stride[0] - 2*padding[0] + (kernel_size[0]-1) + 1)
        out_w = int((in_w-1)*stride[1] - 2*padding[1] + (kernel_size[1]-1) + 1)

        output_size = [out_h, out_w, out_channels]
        text = f"layer_index: {self.layer_index} - type: DeConv. - input-size: {[in_h, in_w, in_channels]} - Output-size: {output_size}."

        return output_size, text

    def calc_maxpooling_layers(self, in_h, in_w, stride):
        # for ker, pad, dil, stri in zip(kernel_size, padding, dilation, stride): # Just for a single layer
        out_h = int((in_h)/stride)
        out_w = int((in_w)/stride)

        output_size = [out_h, out_w]
        text = f"layer_index: {self.layer_index} - type: MaxPool. - input-size: {[in_h, in_w]} - Output-size: {output_size}."

        return output_size, text

    def calc_upsampling_layers(self, in_h, in_w, scale_factor):
        # for ker, pad, dil, stri in zip(kernel_size, padding, dilation, stride): # Just for a single layer
        out_h = int((in_h)*scale_factor)
        out_w = int((in_w)*scale_factor)

        output_size = [out_h, out_w]
        text = f"layer_index: {self.layer_index} - type: UpSamp. - input-size: {[in_h, in_w]} - Output-size: {output_size}."

        return output_size, text
import torch
from new_models.modules import conv2d_bn_relu,deconv_sigmoid,deconv_relu
import numpy as np

class TrajPredictor(torch.nn.Module):
    """ graph convolution network"""
    def __init__(self):
        super(TrajPredictor,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(3,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )
        
        self.deconv_4 = deconv_relu(128,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_4 = torch.nn.Conv2d(128,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        

    def forward(self,x):
        """inputimag: BS,4,3,H,W
        
               x-->conv_stack1 --> conv_stack2 --> conv_stack3 -->  conv_stack4-->              deconv_4     --->           deconv_3        -->       deconv_2            -->deconv_1 --> output
                                                                                     \--> predict_4>up_sample_4---/\--> predict_3>up_sample_3---/\--> predict_2>up_sample_2---/
        """
       
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)

        deconv4_out = self.deconv_4(conv4_out)
        predict_4_out = self.up_sample_4(self.predict_4(conv4_out))

        concat_4 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_4)
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)

        return predict_out


    





        
import torch.nn as nn
import torch.nn.functional as F
import torch
class Multi_stream(nn.Module):
    def __init__(self):
        super(Multi_stream, self).__init__()
        self.ms_convblock1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),\
                                           nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(128),nn.ReLU(inplace=True))
        self.ms_convblock2 =  nn.Sequential(nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),\
                                           nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(256),nn.ReLU(inplace=True))
        self.ms_convblock3 =  nn.Sequential(nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),\
                                           nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(512),nn.ReLU(inplace=True))

    def forward(self, x):
        ms_out1 = self.ms_convblock1(x)
        ms_out2 = self.ms_convblock2(ms_out1)
        ms_out3 = self.ms_convblock3(ms_out2)

        return ms_out3

class Merge_encode(nn.Module):
    def __init__(self):
        super(Merge_encode, self).__init__()

        self.conver1 = nn.Sequential(nn.Conv3d(512, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)), nn.BatchNorm3d(128), nn.ReLU(inplace=True), \
                                           nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),nn.ReLU(inplace=True))
        self.conver2 = nn.Sequential(nn.Conv3d(512, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)), nn.BatchNorm3d(128),nn.ReLU(inplace=True), \
                                           nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),nn.ReLU(inplace=True))
        self.conver3 = nn.Sequential(nn.Conv3d(512, 128,  kernel_size=(3, 3, 3), stride=(1, 1, 1)), nn.BatchNorm3d(128),nn.ReLU(inplace=True), \
                                           nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),nn.ReLU(inplace=True))
        self.conver4 = nn.Sequential(nn.Conv3d(512, 128,  kernel_size=(3, 3, 3), stride=(1, 1, 1)), nn.BatchNorm3d(128),nn.ReLU(inplace=True), \
                                           nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),nn.ReLU(inplace=True))

        self.merge_convblock1 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 1, 1)), nn.ReLU(inplace=True),\
                                           nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(2, 1, 1)), nn.BatchNorm3d(128),nn.ReLU(inplace=True))

    def forward(self, x1,x2,x3,x4):

        x1_c = self.conver1(x1)
        x2_c = self.conver2(x2)
        x3_c = self.conver3(x3)
        x4_c = self.conver4(x4)

        x = torch.cat((x1_c,x2_c,x3_c,x4_c),1)
        ms_out = self.merge_convblock1(x)

        return ms_out
class Merge_decode(nn.Module):
    def __init__(self):
        super(Merge_decode, self).__init__()

        self.mg_convd1 = nn.Sequential(nn.Conv2d(128, 128, 1, 1), nn.ReLU(inplace=True))
        self.mg_convd2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 1, 3, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x, x_size):
        up_depth=[]
        x_d = x[:,:,0,:,:]
        mg_out1 = self.mg_convd1(x_d)
        mg_out2 = self.mg_convd2(mg_out1)

        up_depth.append(F.interpolate(mg_out2, x_size, mode='bilinear', align_corners=True))
     
        return up_depth
# extra part
def extra_layer():
    ms_layer = Multi_stream()
    mgenc_layer = Merge_encode()
    mgdec_layer = Merge_decode()
    return ms_layer, mgenc_layer, mgdec_layer

class Merge_all(nn.Module):
    def __init__(self,ms_layer, mgenc_layer, mgdec_layer):
        super(Merge_all, self).__init__()
        self.multi_stream = ms_layer
        self.merge_encode = mgenc_layer
        self.merge_decode = mgdec_layer
    def forward(self,x1,x2,x3,x4):
        ms_output1 = self.multi_stream(x1)
        ms_output2 = self.multi_stream(x2)
        ms_output3 = self.multi_stream(x3)
        ms_output4 = self.multi_stream(x4)
        x_size = x1.size()[3:]

        ms_output = self.merge_encode(ms_output1,ms_output2,ms_output3,ms_output4)
        up_depth = self.merge_decode(ms_output,x_size)

        return ms_output, up_depth

# build the whole network
def depth_build():
    depth_model = Merge_all(*extra_layer())
    return depth_model

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()







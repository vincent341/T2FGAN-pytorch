#evernote note T2FGAN结构 2020/3/7
import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out

class FusionBlock(torch.nn.Module):
    #fusion+dropout+concat
    def __init__(self,hardstrech=False,dropout=True):
        super(FusionBlock,self).__init__()
        self.hardstretch = hardstrech
        self.mask_sum = 0.0 #sum of mask in the last layer, used to be the loss
        self.drop = torch.nn.Dropout(0.5)
    def forward(self,features_back,features_decoder,features_concat):
        #features_concat are features in encoder
        features_decoder_channel = features_decoder.size(1)
        features_back_channel = features_back.size(1)
        if self.hardstretch:#  processing for the last layer in decoder
            mask = features_decoder[:,-1,:,:]
            self.mask_loss = torch.sum(1.0-self.my_hard_stretch(mask))
            mask_stack = torch.stack((mask,mask,mask),dim=1)
            image_out = features_decoder[:,:-1,:,:]
            output = self.my_hard_stretch(mask_stack)*features_back+image_out*(1.0-self.my_hard_stretch(mask_stack))
            #output = torch.nn.functional.tanh(output)
            return output
        else:
            #fusion
            decoder_firsthalf = features_decoder[:,:features_decoder_channel//2,:,:]
            decoder_lasthalf = features_decoder[:,features_decoder_channel//2:,:,:]
            output_fusion = torch.nn.functional.sigmoid(decoder_firsthalf)*features_back\
                    +decoder_lasthalf*(1-torch.nn.functional.sigmoid(decoder_firsthalf))
            #dropout
            output_dropout = self.drop(output_fusion)
            #concat
            out_concat = torch.cat((output_dropout,features_concat),dim=1)
            return out_concat

    def my_hard_stretch(self,featuremap):
        # map featuremap to [0,1]
        vmin = torch.min(featuremap)
        vmax = torch.max(featuremap)
        value = (featuremap - vmin) / (vmax - vmin + 1e-8)
        return value





class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()
        #Background-encoder
        self.Bconv1 = ConvBlock(input_size=3,output_size=32,activation=False,batch_norm=True)
        self.Bconv2 = ConvBlock(input_size=32,output_size=64,activation=False,batch_norm=True)
        self.Bconv3 = ConvBlock(input_size=64,output_size=128,activation=False,batch_norm=True)
        self.Bconv4 = ConvBlock(input_size=128, output_size=256, activation=False, batch_norm=True)
        self.Bconv5 = ConvBlock(input_size=256, output_size=256, activation=False, batch_norm=True)
        self.Bconv6 = ConvBlock(input_size=256, output_size=256, activation=False, batch_norm=True)
        self.Bconv7 = ConvBlock(input_size=256, output_size=256, activation=False, batch_norm=True)
        
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv1_fdc = FusionBlock(hardstrech=False,dropout=True)#fusion+dropout+concat
        self.deconv2 = DeconvBlock(num_filter * 8//2+num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8//2+num_filter * 8, num_filter * 8, dropout=True)
        self.deconv3_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8//2+num_filter * 8, num_filter * 8)
        self.deconv4_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv5 = DeconvBlock(num_filter * 8//2+num_filter * 8, num_filter * 4)
        self.deconv5_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv6 = DeconvBlock(num_filter * 4//2+num_filter * 4, num_filter * 2)
        self.deconv6_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv7 = DeconvBlock(num_filter * 2//2+num_filter * 2, num_filter)
        self.deconv7_fdc = FusionBlock(hardstrech=False, dropout=True)
        self.deconv8 = DeconvBlock(num_filter//2+num_filter, output_dim, batch_norm=False)
        self.deconv8_fdc = FusionBlock(hardstrech=True, dropout=False)

    def forward(self, x,bx):
        #x:lightmark image bx:background image
        #Background-encoder
        B0 = bx
        B1 = self.Bconv1(B0)#(1,128,128,32)
        B2 = self.Bconv2(B1)#(1,64,64,64)
        B3 = self.Bconv3(B2)#(1,32,32,128)
        B4 = self.Bconv4(B3)#(1,16,16,256)
        B5 = self.Bconv5(B4)#(1,8,8,256)
        B6 = self.Bconv6(B5)#(1,4,4,256)
        B7 = self.Bconv7(B6)#(1,2,2,256)
        # Encoder
        enc0 = x
        enc1 = self.conv1(enc0) #enc1 (1,128,128,64)
        enc2 = self.conv2(enc1) #enc2 (1,64,64,128)
        enc3 = self.conv3(enc2) #(1,32,32,256)
        enc4 = self.conv4(enc3) #(1,16,16,512)
        enc5 = self.conv5(enc4) #(1,8,8,512)
        enc6 = self.conv6(enc5) #(1,4,4,512)
        enc7 = self.conv7(enc6) #(1,2,2,512)
        enc8 = self.conv8(enc7) #(1,1,1,512)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1_fdc = self.deconv1_fdc(B7,dec1,enc7)
        #dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1_fdc)
        dec2_fdc = self.deconv2_fdc(B6,dec2,enc6)
        #dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2_fdc)
        dec3_fdc = self.deconv3_fdc(B5, dec3, enc5)
        #dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3_fdc)
        dec4_fdc = self.deconv4_fdc(B4, dec4, enc4)
        #dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4_fdc)
        dec5_fdc = self.deconv4_fdc(B3, dec5, enc3)
        #dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5_fdc)
        dec6_fdc = self.deconv6_fdc(B2, dec6, enc2)
        #dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6_fdc)
        dec7_fdc = self.deconv7_fdc(B1, dec7, enc1)
        #dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7_fdc)
        self.dec8_out = dec8
        dec8_fdc = self.deconv8_fdc(B0, dec8, enc0)
        out = torch.nn.Tanh()(dec8_fdc)
        self.mask_loss = self.deconv8_fdc.mask_loss
        return out
    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad=False
    def print_param(self):
        for name,param in self.named_parameters():
            print("Param=  ", name, "Requires grad = ",param.requires_grad)


class Generator128(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator128, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv5 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv6 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv7 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc7)
        dec1 = torch.cat([dec1, enc6], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc5], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc4], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc3], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc2], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc1], 1)
        dec7 = self.deconv7(dec6)
        out = torch.nn.Tanh()(dec7)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)


class Discriminator128(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator128, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, stride=1)
        self.conv4 = ConvBlock(num_filter * 4, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
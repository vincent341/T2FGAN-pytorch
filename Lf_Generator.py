import numpy as np
import torchvision,torch
import numpy.matlib

#Light field generator
class Lf_Generator(torch.nn.Module):
    def __init__(self,condition_dict=None,camera=None):
        super(Lf_Generator,self).__init__()
        self.camera = camera #
        """
        self.light_x_v = torch.tensor(condition_dict["lx"],requires_grad=False) #tensor
        self.light_y_v = torch.tensor(condition_dict["ly"],requires_grad=False)
        self.light_z_v = torch.tensor(condition_dict["lz"],requires_grad=False)
        self.belta_R_v = torch.tensor(condition_dict["betar"],requires_grad=True)
        self.belta_G_v = torch.tensor(condition_dict["betag"],requires_grad=True)
        self.belta_B_v = torch.tensor(condition_dict["betab"],requires_grad=True)
        self.light_v = torch.tensor(condition_dict["light"],requires_grad=True)
        self.g_v = torch.tensor(condition_dict['g'],requires_grad=False)

        self.u0_pl = torch.tensor(condition_dict['u0'],requires_grad=False)
        self.v0_pl = torch.tensor(condition_dict['v0'],requires_grad=False)
        self.kx_pl = torch.tensor(condition_dict['kx'],requires_grad=False)
        self.ky_pl = torch.tensor(condition_dict['ky'],requires_grad=False)
"""

        self.light_x_v = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)  # tensor
        self.light_y_v = torch.nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.light_z_v = torch.nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.belta_R_v = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)#0.02
        self.belta_G_v = torch.nn.Parameter(torch.tensor(0.002), requires_grad=True)#0.02
        self.belta_B_v = torch.nn.Parameter(torch.tensor(0.002), requires_grad=True)#0.002
        self.light_v = torch.nn.Parameter(torch.tensor(150.0), requires_grad=False)#original 150.0
        self.g_v = torch.nn.Parameter(torch.tensor(0.6), requires_grad=False)
        """
        self.register_parameter("light_x",self.light_x_v)
        self.register_parameter("light_y", self.light_y_v)
        self.register_parameter("light_z", self.light_z_v)
        self.register_parameter("beta_R",self.belta_R_v)
        self.register_parameter("beta_G", self.belta_G_v)
        self.register_parameter("beta_B", self.belta_B_v)
        self.register_parameter("light_intensity", self.light_v)
        self.register_parameter("g", self.g_v)
        """

        self.u0_pl = torch.tensor(364.5163 / 4.0, requires_grad=False)
        self.v0_pl = torch.tensor(234.3574 / 4.0, requires_grad=False)
        self.kx_pl = torch.tensor(633.5915 / 4.0, requires_grad=False)
        self.ky_pl = torch.tensor(695.2624 / 4.0, requires_grad=False)
        Z_refer = 50
        rescale_factor = 4
        imgwid = int(720)
        imgheight = int(576)
        z_tmp = np.arange(Z_refer, dtype=np.float32)
        z_tmp = np.tile(z_tmp, (imgheight // rescale_factor, imgwid // rescale_factor, 1))
        z_val = np.transpose(z_tmp, (2, 0, 1))
        col_ind = np.arange(imgwid // rescale_factor)  # generate 0,1,2...imgwidth
        col_ind_mat_val = np.matlib.repmat(col_ind, imgheight // rescale_factor, 1)

        row_ind = np.arange(imgheight // rescale_factor)
        row_ind_mat_val = np.transpose(np.matlib.repmat(row_ind, imgwid // rescale_factor, 1))

        self.z_val = torch.tensor(z_val)
        self.col_ind_mat_val = torch.tensor(col_ind_mat_val)
        self.row_ind_mat_val = torch.tensor(row_ind_mat_val)

    def forward(self):
        #g_v[-1,1]
        self.g_v.data = torch.tensor(1.0).data if self.g_v>torch.tensor(1.0) else self.g_v.data
        self.g_v.data = torch.tensor(-1.0) if self.g_v.data < torch.tensor(-1.0) else self.g_v.data
        #belta_R_v >0
        self.belta_R_v.data = torch.tensor(0.0) if self.belta_R_v.data<torch.tensor(0.0) else self.belta_R_v.data
        self.belta_G_v.data = torch.tensor(0.0) if self.belta_G_v.data < torch.tensor(0.0) else self.belta_G_v.data
        self.belta_B_v.data = torch.tensor(0.0) if self.belta_B_v.data < torch.tensor(0.0) else self.belta_B_v.data
        #belta*<1.0 if necessary?
        self.belta_R_v.data = torch.tensor(1.0) if self.belta_R_v.data > torch.tensor(1.0) else self.belta_R_v.data
        self.belta_G_v.data = torch.tensor(1.0) if self.belta_G_v.data > torch.tensor(1.0) else self.belta_G_v.data
        self.belta_B_v.data = torch.tensor(1.0) if self.belta_B_v.data > torch.tensor(1.0) else self.belta_B_v.data

        #belta_R > belta_G and belta_R>belta_B
        self.belta_G_v.data = self.belta_R_v.data.clone() if self.belta_G_v.data>self.belta_R_v.data else self.belta_G_v.data
        self.belta_B_v.data = self.belta_R_v.data.clone() if self.belta_B_v.data > self.belta_R_v.data else self.belta_B_v.data


        x1_mat = (self.col_ind_mat_val - self.u0_pl) * self.z_val / self.kx_pl
        y1_mat = (self.row_ind_mat_val - self.v0_pl) * self.z_val / self.ky_pl
        y1_mat = -y1_mat

        ALdotAO = (x1_mat - self.light_x_v) * x1_mat + (y1_mat - self.light_y_v) * y1_mat + (
                self.z_val - self.light_z_v) * self.z_val  # A is object,O camera,L is light
        AL_len = torch.sqrt((x1_mat - self.light_x_v)**2 + (y1_mat - self.light_y_v)**2 + (self.z_val - self.light_z_v)**2)
        AO_len = torch.sqrt(x1_mat**2 + y1_mat**2 + self.z_val**2)
        # cos = ALdotAO/(AL_len*AO_len+1e-8)
        cos = -ALdotAO / (AL_len * AO_len + 1e-8)  # cos(pi-a)=-cos(a)
        domtmp = 4 * np.pi * pow((1 + self.g_v**2 - 2 * self.g_v * cos), 1.5) + 1e-8
        pg = (1 - self.g_v**2) / domtmp
        t1_R = self.light_v * pg * torch.exp(-self.belta_R_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_R = torch.sum(t1_R, dim=0)

        t1_G = self.light_v * pg * torch.exp(-self.belta_G_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_G = torch.sum(t1_G, dim=0)

        t1_B = self.light_v * pg * torch.exp(-self.belta_B_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_B = torch.sum(t1_B, dim=0)

        BGR_hat = torch.stack([disp6_R, disp6_G, disp6_B], axis=0)#[3,144,180], R,G,B
        BGR_hat_norm = self.min_max_norm(BGR_hat)# normalized to [0,1]
        return BGR_hat_norm
    def min_max_norm(self,x):
        #normalize x by min max (x-min)/(max-min)
        minx = torch.min(x)
        maxx = torch.max(x)
        normx = (x-minx)/(maxx-minx)
        return normx

    def freeze_all(self):
        for name,param in self.named_parameters():
            param.requires_grad=False
    def unfreeze(self,namelist=["belta_G_v","belta_B_v"]):
        for name,param in self.named_parameters():
            if name in namelist:
                param.requires_grad=True
    def print_param(self):
        for name,param in self.named_parameters():
            print("Param=  ", name,"Requires grad = ",param.requires_grad)



"""
if __name__ == "__main__":
    condition_dict = {"lx":0.0,"ly":10.0,"lz":10.0,"betar":0.02,
                      "betag":0.02,"betab":0.002,"light":150.0,
                      "g":0.6,"u0":364.5163 / 4.0,"v0":234.3574/4.0,
                      "kx":633.5915/4.0,"ky":695.2624/4.0}

    gen_bak =Lf_Generator(condition_dict)
    bak_RGB = gen_bak()
    toPIL = torchvision.transforms.ToPILImage()
    toPIL(bak_RGB).show()
"""
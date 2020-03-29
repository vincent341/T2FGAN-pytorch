import numpy as np
import torch,math,torchvision

#Generate light mark image according to given rotation and translation
class LightmarkGenerator(torch.nn.Module):
    def __init__(self):
        #eulermat =[pitch,heading,roll] degree
        #Tmatrix = np.array([[a.Tx, a.Ty, a.Tz]], dtype=np.float32).transpose()
        super(LightmarkGenerator,self).__init__()
        #camera information
        kx = 633.5915
        ky = 695.2624
        u0 = 364.5163
        v0 = 234.3574
        self.Intrin_mat = torch.from_numpy(np.array([[kx, 0, u0], [0, ky, v0], [0, 0, 1]], dtype=np.float32))

        self.imgwid = torch.tensor(int(720),dtype=torch.int,requires_grad=False)
        self.imgheight = torch.tensor(int(576),dtype=torch.int,requires_grad=False)


        #setting of the docking station
        r_station = 1027.0 + 34.0  # mm; r of the docking station + r of lights
        deg_1 = 63.0
        deg_2 = 27.0
        deg_3 = 333.0
        deg_4 = 297.0
        deg_5 = 243.0
        deg_6 = 207.0
        deg_7 = 153.0
        deg_8 = 117.0
        x_light1 = r_station * math.cos(np.deg2rad(deg_1))
        y_light1 = -r_station * math.sin(np.deg2rad(deg_1))  # originally no -

        x_light2 = r_station * math.cos(np.deg2rad(deg_2))
        y_light2 = -r_station * math.sin(np.deg2rad(deg_2))

        x_light3 = r_station * math.cos(np.deg2rad(deg_3))
        y_light3 = -r_station * math.sin(np.deg2rad(deg_3))

        x_light4 = r_station * math.cos(np.deg2rad(deg_4))
        y_light4 = -r_station * math.sin(np.deg2rad(deg_4))

        x_light5 = r_station * math.cos(np.deg2rad(deg_5))
        y_light5 = -r_station * math.sin(np.deg2rad(deg_5))

        x_light6 = r_station * math.cos(np.deg2rad(deg_6))
        y_light6 = -r_station * math.sin(np.deg2rad(deg_6))

        x_light7 = r_station * math.cos(np.deg2rad(deg_7))
        y_light7 = -r_station * math.sin(np.deg2rad(deg_7))

        x_light8 = r_station * math.cos(np.deg2rad(deg_8))
        y_light8 = -r_station * math.sin(np.deg2rad(deg_8))

        wordpoints_array = np.array([[x_light1, x_light2, x_light3, x_light4, x_light5, x_light6, x_light7, x_light8], \
                              [y_light1, y_light2, y_light3, y_light4, y_light5, y_light6, y_light7, y_light8], \
                              [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)  # 8 lights
        Pw = wordpoints_array
        Pw_hom = np.concatenate((Pw, np.ones((1, Pw.shape[1]))), axis=0)
        self.Pw_hom = torch.tensor(Pw_hom)

        self.eulermat = torch.nn.Parameter(torch.tensor(np.array([0.0,0.0,0.0])),requires_grad=True)
        self.Tmatrix = torch.nn.Parameter(torch.tensor(np.array([0.0,0.0,8000.0]).transpose()),requires_grad=True)
        #self.register_parameter("Eulermat", self.eulermat)
        #self.register_parameter("T",self.Tmatrix)

    def eulerAnglesToRotationMatrix(self,theta):
        #convert euler angle to rotation matrix(in tensor)

        R_x = torch.tensor([[1.0, 0.0, 0.0],
                     [0.0, torch.cos(theta[0]), -torch.sin(theta[0])],
                     [0, torch.sin(theta[0]), torch.cos(theta[0])]
                     ])

        R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                     [0, 1, 0],
                     [-torch.sin(theta[1]), 0, torch.cos(theta[1])]
                     ])

        R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                     [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                     [0, 0, 1]
                     ])

        R = torch.mm(R_z, torch.mm(R_y, R_x))

        #numpy to tensor
        R = torch.tensor(R)

        return R

    def deg2rad(self,x):
        pi = torch.tensor(3.1415926535897931)
        x=x*pi/180.0
        return  x

    def forward(self):
        Euler_rad = self.deg2rad(self.eulermat)# tensor
        R_euler = self.eulerAnglesToRotationMatrix(Euler_rad)#tensor
        R = R_euler
        RT = torch.cat((R,self.Tmatrix.unsqueeze(dim=1)),dim=1)
        RT_hom = torch.cat((RT,torch.from_numpy(np.array([[0.0,0.0,0.0,1.0]]))),dim=0)
        Pc_hom = torch.mm(RT_hom,self.Pw_hom)
        Pc = Pc_hom[0:3, :].float()

        Puv_hom = torch.mm(self.Intrin_mat, Pc) / Pc[2, :]
        Puv = Puv_hom[0:2, :]

        #draw point
        lightmark_img = torch.zeros(self.imgheight,self.imgwid,dtype=torch.float32)
        circle_size =3
        for pointi in range(Puv.shape[1]):
            row = Puv[1,pointi].int()
            col = Puv[0, pointi].int()
            row_start,row_end = row-circle_size,row+circle_size
            col_start,col_end = col-circle_size,col+circle_size
            row_start=0 if row_start<0 else row_start #boundrary processing
            row_end = self.imgheight if row_end>self.imgheight else row_end
            col_start = 0 if col_start < 0 else col_start  # boundrary processing
            col_end = self.imgwid if col_end > self.imgwid else col_end

            lightmark_img[row_start:row_end,col_start:col_end]=1.0

        #calculate bbox
        tmp = torch.rand(1)*10
        margin = torch.tensor(10).int()+ tmp.int()# expansion by 10 pixel
        left = torch.min(Puv[0,:]).int()-margin
        right = torch.max(Puv[0,:]).int()+margin
        bottom = torch.max(Puv[1,:]).int()+margin
        top = torch.min(Puv[1,:]).int()-margin
        top =0 if top<0 else top
        bottom=self.imgheight if bottom>self.imgheight else bottom
        left = 0 if left<0 else left
        right = self.imgwid if right>self.imgwid else right
        bbox = [top,bottom,left,right]

        return lightmark_img,bbox

    def freeze_all(self):
        for name,param in self.named_parameters():
            param.requires_grad=False
    def unfreeze(self,namelist=["eulermat","Tmatrix"]):
        for name,param in self.named_parameters():
            if name in namelist:
                param.requires_grad=True
    def print_param(self):
        for name,param in self.named_parameters():
            print("Param=  ", name, "Requires grad = ",param.requires_grad)


if __name__ == "__main__":
    LG =LightmarkGenerator()
    lightmark_img,bbox= LG()#return tensor












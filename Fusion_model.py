import torch,torchvision
import Lf_Generator,LightmarkGenerator
from model import Generator
class Fusion_model(torch.nn.Module):
    #fusion model
    #Pose mark input to MG should be (1,1,256,256), float32
    # light field input to MG should be (1,3,256,256), float32
    def __init__(self,MG_model_path=None):
        super(Fusion_model,self).__init__()
        self.PG = LightmarkGenerator.LightmarkGenerator()
        self.PG.freeze_all()
        self.PG.print_param()
        self.LG = Lf_Generator.Lf_Generator()
        #self.LG.freeze_all()
        self.LG.print_param()
        self.MG = Generator(input_dim=1,num_filter=64,output_dim=4)
        if MG_model_path !=None:
            self.load_MG(MG_model_path)
        self.MG.eval()
        self.MG.freeze_all()

    def forward(self):
        posemark_img, bbox = self.PG()
        posemark_img = torch.unsqueeze(posemark_img,dim=0)
        posemark_img = torch.unsqueeze(posemark_img, dim=0)
        #resize and scale to [-1,1]
        #posemark_img = torchvision.transforms.Resize(posemark_img,(256,256))
        posemark_img = torch.nn.functional.interpolate(posemark_img,(256,256))
        posemark_img = self.preprocess(posemark_img)

        lf_img = self.LG() #RGB
        lf_img = torch.unsqueeze(lf_img,dim=0)
        lf_img = torch.nn.functional.interpolate(lf_img, (256, 256))
        lf_img = self.preprocess(lf_img)
        fusion_img = self.MG(posemark_img,lf_img)
        #torchvision.transforms.ToPILImage()(self.deprocess(fusion_img[0])).show()
        #return  fusion_img
        return fusion_img,lf_img

    def preprocess(self,img):
        return img * 2.0 - 1.0
    def deprocess(self,img):
        return (img+1)/2
    def load_MG(self,model_path):
        #load weights for MG
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path,map_location='gpu')
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        self.MG.load_state_dict(checkpoint['state_dict_G'])

    def freeze_all(self):
        self.LG.freeze_all()
        self.PG.freeze_all()
        self.MG.freeze_all()
    def unfreeze(self,LG_unfreeze=True,PG_unfreeze=True,namelist=["belta_G_v","belta_B_v","eulermat","Tmatrix"]):
        if LG_unfreeze:
            self.LG.unfreeze(namelist=namelist)
        if PG_unfreeze:
            self.PG.unfreeze(namelist=namelist)
    def print_param(self):
        self.LG.print_param()
        self.PG.print_param()
        self.MG.print_param()

if __name__ == "__main__":
    MG_model_path = "D:\\MyProgram\\pythonprogram\\t2fgan_pytorch20200303\\mygai\\checkpoint\\checkpoint-tank-rc-mask-200316-150.pth"
    #load master generator
    fusion_model = Fusion_model(MG_model_path)
    fusion_img = fusion_model()
    fusion_img = fusion_model.deprocess(fusion_img)

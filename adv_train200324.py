import torch,torchvision,argparse,os,tensorboardX,random
from Myclassifier import Myclassifier
from Fusion_model import Fusion_model
from LightmarkGenerator import LightmarkGenerator
from Lf_Generator import Lf_Generator


class Ensemble_Model(torch.nn.Module):
    def __init__(self,fusionmodel,classifier):
        super(Ensemble_Model,self).__init__()
        self.FM = fusionmodel
        self.classifier = classifier
    def forward(self):
        self.img,self.lf_img = self.FM()
        input_img = torch.cat([self.img,])

        if random.random() > 0.5:
            input_classifier = torch.cat([self.img, self.lf_img],dim=0)  # input foreground and background image
            label_real = torch.tensor([1, 0])
        else:
            input_classifier = torch.cat([self.lf_img, self.img], dim=0)
            label_real = torch.tensor([0, 1])

        pred_label = self.classifier(input_classifier)#classification result for foreground image
        return pred_label,label_real



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpath', type=str, default="checkpoint-adv/", help='checkpoint path')
    parser.add_argument('--logpath', type=str, default="log-adv/", help='log path')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--lrFM', type=float, default=0.1, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrC', type=float, default=0.0001, help='learning rate for generator, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--saveepoch', type=int, default=50, help='save model every n epochs')
    parser.add_argument('--MGpath', type=str, default="D:\\MyProgram\\pythonprogram\\t2fgan_pytorch20200303\\mygai\\checkpoint\\checkpoint-tank-rc-mask-200316-150.pth", help='path of mg model')
    parser.add_argument('--Cpath', type=str,default="D:\\MyProgram\\pythonprogram\\t2fgan_pytorch20200303\\mygai\\checkpoint-c\\40.pth",
                        help='path of classifier model')
    params = parser.parse_args()
    print(params)

    if not os.path.exists(params.checkpath):
        os.mkdir(params.checkpath)
    if not os.path.exists(params.logpath):
        os.mkdir(params.logpath)

    use_cuda = torch.cuda.is_available()

    #Define model
    fusion_model = Fusion_model(MG_model_path=params.MGpath)#load pretrain model
    classifier = Myclassifier(model_path=params.Cpath)
    ensemble_model = Ensemble_Model(fusion_model,classifier)

    if use_cuda:
        #fusion_model = fusion_model.cuda()
        #classifier = classifier.cuda()
        ensemble_model =ensemble_model.cuda()
    #Define optimizer
    #optimizer = torch.optim.Adam(ensemble_model.parameters(),lr=params.lr, betas=(params.beta1, params.beta2))
    #optimizer_FM = torch.optim.Adam(ensemble_model.FM.parameters(), lr=params.lrFM, betas=(params.beta1, params.beta2))
    #optimizer_FM = torch.optim.Adam([ensemble_model.FM.LG.belta_G_v,ensemble_model.FM.LG.belta_B_v], lr=params.lrFM, betas=(params.beta1, params.beta2))
    optimizer_FM = torch.optim.SGD(ensemble_model.FM.parameters(), lr=params.lrFM, momentum=0.9)
    optimizer_C = torch.optim.Adam(ensemble_model.classifier.parameters(), lr=params.lrC, betas=(params.beta1, params.beta2))
    #optimizer = torch.optim.SGD(ensemble_model.parameters(), lr=params.lr, momentum=0.9)
    #Define Loss
    C_loss_func = torch.nn.CrossEntropyLoss()
    #log writer
    writer_train = tensorboardX.SummaryWriter(log_dir=params.logpath)
    start_epoch = 0

    for epoch in range(start_epoch, params.num_epochs):
        #first freeze classifier, train ensemble model============================================
        ensemble_model.FM.unfreeze(namelist=["belta_G_v","belta_B_v","belta_R_v"])
        ensemble_model.classifier.freeze_all()

        for k in ensemble_model.FM.LG.parameters():
            k.retain_grad()
        for k in ensemble_model.FM.PG.parameters():
            k.retain_grad()

        if epoch==0:
            print("After freezing classifier================")
            ensemble_model.classifier.print_param()
            ensemble_model.FM.print_param()
        pred_label,label_real = ensemble_model()#!gai
        gen_img = ensemble_model.FM.deprocess(ensemble_model.img)#show generated img
        #torchvision.transforms.ToPILImage()(gen_img[0]).show()

        # 1 for foreground, 0 for background
        c_loss = C_loss_func(pred_label, label_real)#Input: (N,C) where C = number of classes;Target: (N) where each value is 0 <= targets[i] <= C-1
        if label_real[0].data ==1: #first one is generated(foreground image)
            gen_loss = C_loss_func(pred_label[0, :].unsqueeze(dim=0), torch.LongTensor([label_real[0]]))
            lf_loss = C_loss_func(pred_label[1, :].unsqueeze(dim=0), torch.LongTensor([label_real[1]]))
        else:
            lf_loss = C_loss_func(pred_label[0, :].unsqueeze(dim=0), torch.LongTensor([label_real[0]]))
            gen_loss = C_loss_func(pred_label[1, :].unsqueeze(dim=0), torch.LongTensor([label_real[1]]))
        FM_loss = -c_loss
        print("Epoch=",epoch,"FM_loss=",FM_loss.item())

        #record for first half epoch 2*epoch
        writer_train.add_scalar(tag="FM loss",scalar_value=FM_loss.item(),global_step=epoch*2)
        writer_train.add_scalar(tag="Gen classify loss", scalar_value=gen_loss.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="Backgr classify loss", scalar_value=lf_loss.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="light_v", scalar_value=ensemble_model.FM.LG.light_v.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="beta_G", scalar_value=ensemble_model.FM.LG.belta_G_v.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="beta_B", scalar_value=ensemble_model.FM.LG.belta_B_v.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="beta_R", scalar_value=ensemble_model.FM.LG.belta_R_v.item(), global_step=epoch*2)
        writer_train.add_scalar(tag="classify loss", scalar_value=c_loss.item(), global_step=epoch*2)
        #writer_train.add_scalar(tag="light_v grad", scalar_value=ensemble_model.FM.LG.light_v.grad.item(), global_step=epoch)
        #writer_train.add_scalar(tag="beta_R grad", scalar_value=ensemble_model.FM.LG.belta_R_v.grad.item(),global_step=epoch)

        if epoch%10==0:
            #writer_train.add_image(tag="Generated img", img_tensor=gen_img[0], global_step=epoch)
            writer_train.add_image(tag="Generated img", img_tensor=gen_img[0], global_step=epoch*2)

        # backward update
        ensemble_model.FM.zero_grad()
        FM_loss.backward(retain_graph=True)
        optimizer_FM.step()

        writer_train.add_scalar(tag="beta_G grad", scalar_value=ensemble_model.FM.LG.belta_G_v.grad.item(),global_step=epoch*2)
        writer_train.add_scalar(tag="beta_B grad", scalar_value=ensemble_model.FM.LG.belta_B_v.grad.item(),global_step=epoch*2)
        #unfreeze  classifier , freeze \theta, update Classifier=================================================

        ensemble_model.classifier.unfreeze()
        ensemble_model.FM.freeze_all()
        if epoch==0:
            print("After unfreezing classifier, freezing FM================")
            ensemble_model.classifier.print_param()
            ensemble_model.FM.print_param()

        pred_label, label_real = ensemble_model()#generate image by using current FM(\theta)
        gen_img = ensemble_model.FM.deprocess(ensemble_model.img)  # show generated img
        # torchvision.transforms.ToPILImage()(gen_img[0]).show()
        # 1 for foreground, 0 for background
        c_loss = C_loss_func(pred_label, label_real)
        #record for first half epoch 2*epoch
        writer_train.add_scalar(tag="FM loss",scalar_value=FM_loss.item(),global_step=epoch*2+1)
        writer_train.add_scalar(tag="Gen classify loss", scalar_value=gen_loss.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="Backgr classify loss", scalar_value=lf_loss.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="light_v", scalar_value=ensemble_model.FM.LG.light_v.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="beta_G", scalar_value=ensemble_model.FM.LG.belta_G_v.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="beta_B", scalar_value=ensemble_model.FM.LG.belta_B_v.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="beta_R", scalar_value=ensemble_model.FM.LG.belta_R_v.item(), global_step=epoch*2+1)
        writer_train.add_scalar(tag="classify loss", scalar_value=c_loss.item(), global_step=epoch*2+1)

        ensemble_model.classifier.zero_grad()
        c_loss.backward()
        optimizer_C.step()



        """
        ensemble_model.FM.freeze_all()
        ensemble_model.FM.print_param()
        pred_label = ensemble_model()
        c_loss = C_loss_func(pred_label,1)
        """
    writer_train.close()



import torch,torchvision
import collections,tensorboardX,tqdm,time,shutil,glob,os
import argparse
import numpy as np
import torch.utils.data as Data
from prefetch_generator import BackgroundGenerator

def load_ckp(checkpoint_fpath, model, optimizer):
    #latest checkpoint
    check_list = glob.glob(checkpoint_fpath+"*.pth")
    if check_list==[]:
        return model,optimizer, 0,0
    check_epochs = [int(os.path.basename(x)[:-4]) for x in check_list]
    max_epoch = max(check_epochs)
    checkpoint = torch.load(checkpoint_fpath+str(max_epoch)+'.pth')

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model,optimizer, checkpoint['epoch'], checkpoint['step']
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path+str(epoch)+".pth"
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def freeze_convnet(model):
    for name,module in model.named_modules():
        if name!='fc_layer':
            for param in module.parameters():
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = True
    return model

def _channels(img):
    if len(img.split())!=3:
        temp = np.zeros((img.shape[0], img.shape[1], 3))
        temp[:, :, 0] = img
        temp[:, :, 1] = img
        temp[:, :, 2] = img
        img = temp
        img = img.transpose(img, (2, 0, 1))  ## reshape
    return img
def _M2N1(img):
    return img*2.0-1.0

class Myclassifier(torch.nn.Module):
    def __init__(self,model_path=None):
        super(Myclassifier, self).__init__()
        resnetmodel = torchvision.models.resnet18(pretrained=True)
        self.num_ftrs = resnetmodel.fc.in_features
        self.resnet_layer = torch.nn.Sequential(collections.OrderedDict(list(resnetmodel.named_children())[:-1]))
        #resnetmodel = torchvision.models.mobilenet_v2(pretrained=True)
        ##self.resnet_layer = resnetmodel.features
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        #self.fc_layer = nn.Linear(in_features=self.num_ftrs,out_features=100,bias=True)
        self.add_module('fc_layer', torch.nn.Linear(in_features=self.num_ftrs, out_features=2, bias=True))
        #self.resnetmodel = torchvision.models.squeezenet1_1(pretrained=True)
        #self.resnetmodel.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        if model_path != None:
            self.load_model(model_path)

    def forward(self,x):
        x = self.resnet_layer(x)
        # x=self.fc_layer(x.view(x.size(0),-1))
        x = self.fc_layer(x.view(x.size(0), -1))
        #x = self.resnetmodel(x)
        return x
    def freeze_all(self):
        for name,param in self.named_parameters():
            param.requires_grad=False
    def unfreeze(self):
        for name,param in self.named_parameters():
            param.requires_grad=True

    def load_model(self,model_path):
        #load weights for MG
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path,map_location='gpu')
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
    def print_param(self):
        for name,param in self.named_parameters():
            print("Param=  ", name, "Requires grad = ",param.requires_grad)


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument("-b","--batch",required=False,default=16)
    parser.add_argument("-lr","--learningrate",required=False,default=1e-2)
    parser.add_argument("-e", "--epoch", required=False,default=100)
    parser.add_argument("-ld", "--logdir", required=False, default="log-c/")
    parser.add_argument('--checkpath', type=str, default="checkpoint-c/", help='checkpoint path')
    parser.add_argument('--resume', type=bool, default=True, help='Resume from checkpoint')
    opt=parser.parse_args()
    if not os.path.exists(opt.checkpath):
        os.mkdir(opt.checkpath)
    #torch.manual_seed(0)
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.Lambda(lambda img: _channels(img)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda img: _M2N1(img)),
        #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #dataset = torchvision.datasets.ImageFolder(root='./cat-dog/',transform=data_transform)

    train_dataset = torchvision.datasets.ImageFolder(root='D:\\MyProgram\\pythonprogram\\Classification_adversial190705\\dataset_classify_adver190707\\train\\',
                                                     transform=data_transform)

    test_dataset = torchvision.datasets.ImageFolder(
        root='D:\\MyProgram\\pythonprogram\\Classification_adversial190705\\dataset_classify_adver190707\\val_field\\',
        transform=data_transform)

    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #train_dataset,val_dataset = train_valid_split(dataset)
   #print(train_dataset.classes,dataset.class_to_idx)
    #print(dataset.imgs)
    #print(train_dataset[0][0].shape)
    #print(train_dataset[0][1])
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=int(opt.batch), shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset,batch_size=int(opt.batch),shuffle=True)
    dataset_sizes={'train':len(train_dataset),'val':len(test_dataset)}

    #tensorboardX
    writer_train = tensorboardX.SummaryWriter(log_dir=opt.logdir+"train/")
    writer_test = tensorboardX.SummaryWriter(log_dir=opt.logdir + "test/")
    #network-------------------------------------------------------------------

    mynet = Myclassifier()
    print(mynet)
    #freeze other than fc layer
    #freeze_convnet(mynet)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mynet.parameters(),lr=float(opt.learningrate))
    #GPU setting
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        mynet=mynet.cuda()
    #load checkpoints
    if opt.resume:
        mynet, optimizer,start_epoch,step = load_ckp(opt.checkpath,mynet,optimizer)
        start_epoch+=1
    n_iter = 0
    for epoch in range(start_epoch,opt.epoch):
        loss_train = 0
        loss_val = 0
        train_corrects_step = 0
        val_corrects_step = 0
        pbar = tqdm.tqdm(enumerate(BackgroundGenerator(train_loader)),total = len(train_loader))
        start_time = time.time()

        #for step,(img_data,img_label) in enumerate(train_loader):
        mynet.train()  #
        train_step =0
        for step,data in pbar:
            n_iter+=1
            train_step+=1
            img_data,img_label = data #1 for foreground, 0 for background
            if use_cuda:
                img_data = img_data.cuda()
                img_label = img_label.cuda()

            prepare_time = start_time-time.time()

            #count = count+1
            #print(img_data.shape)
            #print(img_label)

            #print("label", img_label)
            #train mode===================================================================

            #print("fc grad",mynet.fc_layer.weight.grad,"require grad:",mynet.fc_layer.weight.requires_grad)
            out = mynet(img_data)
            _,preds = torch.max(out,1)
            loss = loss_fn(out,img_label)
            optimizer.zero_grad()
            loss.backward()
            #print("fc grad", mynet.fc_layer.weight.grad, "require grad:", mynet.fc_layer.weight.requires_grad)
            optimizer.step()
            loss_train += loss.item()*img_data.size(0)
            train_corrects_step +=torch.sum(preds==img_label.data)
            process_time = start_time-prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, opt.epoch))
            start_time = time.time()
            #index = torch.max(out,dim=1)[1].numpy()[0]
            #tmp = img_data[0,:,:,:].numpy()
            #show_img =Image.fromarray(tmp.astype('uint8')).convert('RGB')
            #plt.figure('first img')
            #plt.imshow(np.transpose(tmp,(1,2,0)))
            #plt.show()
            #print("index",index)
            #print("step：",step,"loss:",loss.data.numpy())
        #record train loss
        writer_train.add_scalar(tag='loss', scalar_value=loss_train/dataset_sizes['train'], global_step=epoch)
        writer_train.add_scalar(tag='acc', scalar_value=float(train_corrects_step.item()) / dataset_sizes['train'], global_step=epoch)

        #val loss
        mynet.eval()
        val_step = 0
        for step, (img_data_val, img_label_val) in enumerate(test_loader):
            val_step+=1
            if use_cuda:
                img_data_val = img_data_val.cuda()
                img_label_val = img_label_val.cuda()
            with torch.no_grad():
                out_val = mynet(img_data_val)
                _, preds = torch.max(out_val, 1)
                val_corrects_step += torch.sum(preds == img_label_val.data)
                loss = loss_fn(out_val, img_label_val)
                loss_val+=loss.item()*img_data_val.size(0)

        writer_test.add_scalar(tag='loss', scalar_value=loss_val/dataset_sizes['val'], global_step=epoch)
        writer_test.add_scalar(tag='acc', scalar_value=float(val_corrects_step.item()) / dataset_sizes['val'], global_step=epoch)
        if epoch%5==0:
            print('Epoch {}, Training loss {}'.format(epoch,float(loss_train/train_step)))
            print('Epoch {}, Test loss {}'.format(epoch, float(loss_val/ val_step)))
            checkpoint = {
                'step': step + 1,
                'epoch': epoch + 1,
                'state_dict': mynet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_ckp(checkpoint, False, opt.checkpath, opt.checkpath)




    writer_train.close()
    writer_test.close()
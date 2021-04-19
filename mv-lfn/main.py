from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

# from data_utils.getDataset import GetDataTrain
from data_utils.getDataset_modelnet2 import GetDataTrain

from model.myresnet import myresnet
#from model.triCenter_loss import TripletCenterLoss
#from model.center_loss import CenterLoss
#from model.cosCenter_loss import CenterLoss as CosLoss

from scipy.io import savemat
import itertools

'''

'''

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch',  default=80, type=int, help='number of epoch in training')
    parser.add_argument('--j',  default=4, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
    parser.add_argument('--pretrained', dest='pretrained', action ='store_true', help='use pre-trained model')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--stage', type=str, default='train', help='train test, extract feature')
    parser.add_argument('--views', default=9, type=int, help='the number of views')
    parser.add_argument('--num_classes',  default=40, type=int, help='the number of clsses')
    parser.add_argument('--rs',  default = 16, type = int, help='For single view')
    parser.add_argument('--rm',  default = 0, type = int, help='Among multiple view fusion, rm = sqrt(views)')
    parser.add_argument('--model_name', type=str, default='subpixel_double', help='train test')

    return parser.parse_args()


args = parse_args()
args.device = torch.device('cuda:%s'%args.gpu)
# 可以针对全局，也可以针对局部
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# use: 输入optimizer 和 epoch 就可以使用
#
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * (0.1 ** (epoch*3 //args.epoch))   # 每两百个
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print ('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))
    return lr

def dist_elur(fts_q, fts_c):
    fts_q = fts_q/torch.norm(fts_q, dim=-1,keepdim=True)
    fts_c = fts_c/torch.norm(fts_c, dim=-1,keepdim=True)
    fts_qs = torch.sum((fts_q)**2,dim=-1,keepdim=True)
    fts_cs = torch.sum((fts_c)**2,dim=-1,keepdim=True).t()
    qc = torch.mm(fts_q,fts_c.t())
    dist = fts_qs + fts_cs - 2 * qc +1e-4
    return torch.sqrt(dist)

def dist_cos(fts_q, fts_c):
    up = torch.matmul(fts_q,fts_c.T)
    down1 = torch.sqrt(torch.sum((fts_q)**2,axis=-1,keepdims=True))
    down2  = torch.sqrt(torch.sum((fts_c)**2,axis=-1,keepdims=True).t())
    down = torch.mm(down1, down2)
    dist = up/(down+1e-4)
    return 1-dist


def mi_dist(fts1,fts2,la1,la2,margin=1,mode=2):
    dist = dist_elur(fts1,fts2)
    index = (la1[:,mode].reshape(-1,1)==la2[:,mode].reshape(1,-1)).bool()
    ap = dist[index]
    lens = len(ap)
    an = torch.sort(dist[(1-index.long()).bool()])[0][:lens]
    if lens*2 > (dist.shape[0]**2):
        ap = ap.mean().unsqueeze(0)
        an = an.mean().unsqueeze(0)
    loss = nn.MarginRankingLoss(margin)(ap,an,torch.Tensor([-1]).to(device)    )
    return loss,ap.mean()


path_model = [
    'experiment/checkpoints/top.pth' # 0   训练最好的
]

path_mat =[os.path.join('metric', os.path.basename(i).split('pth')[0] + 'mat') for i in path_model]

def g_t(num, target):
    tx = torch.ones(target.shape[0],num, num) 
    tx = (tx * args.num_classes).long()
    for i in range(num):
        tx[:, i, i] = target
    return tx



def main():
    global args
    # 数据记录
    logger_train = get_logger('%s_train'%(args.model_name))
    logger_test = get_logger('%s_test'%(args.model_name))
    top_acc = 0.0
    top_acc_path = ''
    acc_avg = AverageMeter()
    losses = []

    # domainMode 0,1  是image, render/ mask  + train/test
    #           2,3   是image, render, mask  + 全部数据
    #           4,5   是image /render,      + train/test
    trainDataset =  GetDataTrain(dataType='train', imageMode='RGB', views=args.views)
    validateDataset =  GetDataTrain(dataType='test', imageMode='RGB', views=args.views)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size =args.batchsize,shuffle=True,num_workers=args.j, pin_memory = True, drop_last=True)
    validateLoader = torch.utils.data.DataLoader(validateDataset, batch_size =args.batchsize, shuffle=False, num_workers=args.j,pin_memory=True,  drop_last=False)
    
    

    model = myresnet(args=args)
   
    # process: gpu use
    if args.gpu == '0,1':
        device_ids = [int(x) for x in args.gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif args.gpu == '0' or args.gpu == '1':
        #device = torch.device('cuda:'+args.gpu)
        #model.to(device)
        model.to(args.device)
        #cri_triCenter_d.to(args.device)
        #cri_triCenter_c.to(args.device)
        
        # process: optimizer
    if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            #optimizer_triCenter_c = torch.optim.SGD(itertools.chain(cri_triCenter_cat.parameters(),cri_triCenter_cent.parameters()), lr=0.1)
    elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.wd
            )
            
    path_load =   'experiment/checkpoints/epoch_63_acc_0.9677.pth'
    if args.stage=='exft':
        print('Extract feature ...')
        extract_feat(args,  model.eval(), path_load, 'metric/model_feat.npy', validateLoader)
        exit(-1)
    elif args.stage == 'test':
        print('Stage:test...')
        model.load_state_dict(torch.load(path_load))
        loss,acc = Validate(args, model.eval(), validateLoader)
        print(f'---loss:{loss}, acc:{acc[-1]}')
        exit(-1)
    else: 
        if args.pretrained:
            print('Pretrianed...')
            model.load_state_dict(torch.load(path_load))
        print('Train...')
 
    # train
    for epoch in range(0, args.epoch):
        cur_lr = adjust_learning_rate(optimizer,epoch)
        ftsa = []
        laa = []
        acc_avg.reset()
        
        for idx, input_data in enumerate(tqdm(trainLoader)):
            data = input_data['data'].to(args.device)
            target = input_data['target'].reshape(-1)
            #tx = g_t(args.views, target).to(args.device)
            target = target.to(args.device)
            model.train()
            out,fts= model(data)
            
            # out = out.reshape(-1, args.num_classes + 1)
            loss = F.cross_entropy(out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
             
            acc = get_acc_topk(out.cpu().data, target.cpu().data)
            acc_avg.update(acc) 

            if (idx+1) % 100 ==0:
                print_loss = 'epoch:%d, loess:%.4f'%(epoch, loss)
                for i in losses:
                    print_loss += ', loss%s : %.4f'%(i, eval('loss'+i))
                print(print_loss)

                logger_train.info(print_loss)
            #if idx==10:break;
        # 作用：测试并保存数据
        if (epoch+1) % 1 == 0:
            loss,acc = Validate(args, model.eval(), validateLoader)
            print('loss',loss, acc)
            logger_test.info('---save model epoch:%d, acc:%.5f' % (epoch, acc[-1]))
            if acc[-1] > top_acc:
                top_acc = acc[-1]
                print('save model...')
                if top_acc_path !='':    # 删去之前的包，这里可以不适用
                    os.remove(top_acc_path)
                # top_acc_path = save_model(model, model_name, epoch, top_acc)
                top_acc_path = save_model(model, args.model_name, epoch, top_acc, top=True)
        if (epoch+1) == args.epoch:
            top_acc_path = save_model(model, args.model_name, epoch, acc[-1], top=False)

def save_model(model,model_name,epoch,acc, top=True):
    checkpoints = 'experiment/checkpoints'
    print('Save model epoch:%d, acc:%.3f ... '% (epoch, acc))
    fs = os.path.join(checkpoints,'%s_epoch_%d_acc_%.4f.pth'%(model_name,epoch,acc))
    torch.save(model.state_dict(),fs)
    if top:
        torch.save(model.state_dict(), 'experiment/checkpoints/%s_top.pth'%model_name)
        print('Save model of top acc ...' )
    return fs


def load_model(model,path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)


def get_acc_of_out(out,target):
    choice = out.max(1)[1]
    correct = choice.eq(target.long()).sum()
    return correct.item() / float(len(target))

def get_acc_topk(out,target,topk=(1,)):
    batch_size = target.shape[0]
    topkm = max(topk)
    _, pred = out.topk(topkm, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(1.0 / batch_size))
    return np.array(acc)

def extract_feat(args, model, path_load, path_save, data):
    pretrained = torch.load(path_load)
    model.load_state_dict(pretrained)
    ftss = []
    lass = []
    names = []
    for idx, intput_data in enumerate(tqdm(data)):
        data   = intput_data['data'].to(args.device)
        target = intput_data['target'].reshape(-1)  #.to(args.device)
        # name = intput_data['name'].reshape(-1)  #.to(args.device)
        with torch.no_grad():
            out,fts = model(data)

        ftss.append(fts.cpu().data)
        lass.append(target.cpu().data)
        #names.append(name)
    ftss = torch.cat(ftss, dim=0).numpy()
    lass = torch.cat(lass, dim=0).numpy()
    # names = np.concatenate(names, axis=0)
    return np.save(path_save, {'fts':ftss, 'las':lass})



def Validate(args, model, validateLoader):
    # 各类准确率
    acc_avg = AverageMeter()
    loss_avg = AverageMeter()
    
    for idx, intput_data in enumerate(tqdm(validateLoader)):
        data = intput_data['data'].to(args.device)
        target = intput_data['target']#.to(args.device)
        
        with torch.no_grad():
            out,_ = model(data)
        
        out= out.cpu().data
        # loss = nn.CrossEntropyLoss()(out, target)
        acc = get_acc_topk(out, target, (1,))
        
        acc_avg.update(np.array([acc]).reshape(-1))
        #loss_avg.update(torch.Tensor([loss0,loss2]))
        #if idx==10: break;

    return 0, acc_avg.avg

def test(model, test_loader):
    batch_correct =[]
    batch_loss =[]
    
   
    for batch_idx, (data, target) in enumerate(test_loader):
        pred,y = model(data.view(-1,784))
        loss = (pred, target)
        choice = pred.data.max(1)[1]
        correct = choice.eq(target.long()).sum()
        batch_correct.append(correct.item()/float(len(target)))
        batch_loss.append(loss.data.item())
    # print('test:',np.mean(batch_correct))
    return np.mean(batch_correct), np.mean(batch_loss)



if __name__ == '__main__':
    main()

    # img = GetDataTrain(dataType='train', imageMode='RGB', domainMode=0)


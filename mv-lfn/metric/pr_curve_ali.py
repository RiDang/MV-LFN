import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from pr_curve import path1,path2
import json

path1_ = [
'orth_L1_dist_epoch_99_acc_0.5607.matimgtest',# 0
'single_center_p_mi_cos_epoch_19_acc_0.8852.matimgtest',
'mv_pn5_epoch_19_acc_0.8576.mattestimg', # 2
'cos_mi1_epoch_19_acc_0.8804.mattestimg', # 3
'mv_pn5_mi_epoch_39_acc_0.0000.mattestimg', # 4
'mv_pn5_mi1_epoch_9_acc_0.0000.mattestimg', # 5
'save_mv_pn5_mi_epoch_39_acc_0.0000.mattestimg', # 6
'save_mv_pn5_mi1_epoch_9_acc_0.0000.mattestimg', # 7
]

path2_ = [
'orth_L1_dist_epoch_99_acc_0.5607.matrendtest',
'single_center_p_mi_cos_epoch_19_acc_0.8852.matrendtest',
'mv_pn5_epoch_19_acc_0.8576.mattestrend', # 2
'cos_mi1_epoch_19_acc_0.8804.mattestrend', # 3
'mv_pn5_mi_epoch_39_acc_0.0000.mattestrend', # 4
'mv_pn5_mi1_epoch_9_acc_0.0000.mattestrend', # 5
'save_mv_pn5_mi_epoch_39_acc_0.0000.mattestrend', # 66
'save_mv_pn5_mi1_epoch_9_acc_0.0000.mattestrend', # 7
]

index =7
name = 'test_naive_model_gray_cos'
p_q = path1_[index]
p_c = path2_[index]
#p_c = 'best_feat.mat'


# fts_q = np.random.randn(100,128)  # feature qurey
# las_q = np.random.randint(0,9,100)  # label
# fts_c = np.random.randn(100,128)  # feature contrain
# las_c = np.random.randint(0,9,100)  # label



def get_data(p_q, p_c):
    a = loadmat(p_q)
    b = loadmat(p_c)
    print('a keys:', a.keys())
    fts_q = a['fts']  # feature qurey
    las_q = a['las']  # label
    fts_c = b['fts']  # feature contrain
    #las_c = b['las'][:,1]  # label
    las_c_a = b['las']  # label
    print(fts_q.shape, fts_c.shape)
    #las_q = np.array(las_q).reshape(-1)  # 规范标签的形状
    #las_c = np.array(las_c).reshape(-1)
    #print(las_q[-100:]);exit(-1)
    return fts_q, fts_c, las_c_a
    #return fts_q,las_q, fts_c,las_c, las_c_a

# 输入q,c batch x feature
# 输出：q * c  --query x contain
def dist_euler(fts_q, fts_c):
    fts_qs = np.sum(np.square(fts_q),axis=-1,keepdims=True)
    fts_cs = np.sum(np.square(fts_c),axis=-1,keepdims=True).T
    qc = np.matmul(fts_q,fts_c.T)
    dist = fts_qs + fts_cs - 2 * qc
    return dist

def dist_cos(fts_q, fts_c):
    
    up = np.matmul(fts_q,fts_c.T)
    down1 = np.sqrt(np.sum(np.square(fts_q),axis=-1,keepdims=True))
    down2  = np.sqrt(np.sum(np.square(fts_c),axis=-1,keepdims=True).T)
    down = np.matmul(down1, down2)
    dist = up/(down+1e-4)
    return 1-dist


def get_pr(dist_p=0):
    fts_q, fts_c,las_c_a = get_data(p_q, p_c)
    if dist_p==0:
        dist = dist_euler(fts_q, fts_c)
    else:dist = dist_cos(fts_q, fts_c)
    len_q,len_c = dist.shape
   
    #model_label = json.load(open('/home/dh/zdd/data/ali_system/information/model_label.json','r'))
    model = np.unique(np.sort(las_c_a[:,-1]))
    len_c_a = len(model)
    model_mask = np.tile(model,(len_c,1)).T
    model_a = np.tile(las_c_a[:,-1],(len_c_a,1))  # 横向的
    model_mask = (model_mask ==model_a )
    model_mask = model_mask/(model_mask.sum(-1,keepdims=True))
    dist_m = np.zeros((len_q, len_c_a)) 
    dist = np.matmul(dist,model_mask.T)
    #las_c =np.array([model_label['%07d'%i] for i in model])[:,1]
    las_c = model #las_c_a[:,-1]
    fs = open('retrieval_results_elur.txt','w+') if dist_p==0 else open('retrieval_results_cos.txt','w+')
    for i in tqdm(range(len_q)):
        data = ','.join(np.round(dist[i,:],decimals=3).astype(str).tolist())
        #print(data+i)
        fs.writelines([':'.join([('%07d'%i),data])])
        fs.write('\n')
    fs.close()
        
    #dist_index = np.argsort(dist, axis=-1)
    #dist = np.sort(dist, axis=-1)
    
    #len_c = len_c_a
    # 利用标签计算，标记检索结果
    #result = np.zeros_like(dist)
    #laq_bool = np.tile(las_q,(len_c,1)).T
    #lac_bool = np.tile(las_c,(len_q,1))  # 需要对c 的标签进一步排序
    #index = np.tile(np.array(range(len_q)),(len_c,1)).T
    #lac_bool = lac_bool[index,dist_index]
#    result = (laq_bool == lac_bool)

#    p = np.zeros(len_c)
#    r = np.zeros(len_c)
#    r_all = np.sum(result)
#    for i in tqdm(range(len_c)):
#        s = np.sum(result[:,:i+1])
#        p[i] = s/((i+1)*len_q)
#        r[i] = s/r_all
#    mAP = np.sum((r[1:] - r[:-1])*p[:-1])
#    fo =open('pr/%s_%s_%.5f.txt' %(name, 'cos' if dist_p else 'elur', mAP),'w') 
    #np.savetxt(fo, np.array([r, p]), fmt='%.5f')
    #fo.close() 
    #return np.array([r, p]),mAP
    return 0,0 


if __name__ == '__main__':

    pr,mAP = get_pr(0)
    print('elur map:',mAP) 
    pr,mAP = get_pr(1)
    print('cos map:',mAP) 
    
    # print('pr:',pr.shape)
    # plt.plot(pr[0,:],pr[1,:])
    # plt.title('map:%f'%map)
    # plt.show()

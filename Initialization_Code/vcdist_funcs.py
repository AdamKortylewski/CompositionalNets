import numpy as np

def vc_dis(inst1, inst2, deform):
    inst1 = inst1[:,:,0:14]
    inst2 = inst2[:,:,0:14]
    hh1 = inst1.shape[1]
    hh2 = inst2.shape[1]
    if hh1 > hh2:
        diff = hh1 - hh2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.concatenate([np.zeros((inst2.shape[0], diff_top, inst2.shape[2])), inst2], axis=1)
        inst2 = np.concatenate([inst2, np.zeros((inst2.shape[0], diff_bottom, inst2.shape[2]))], axis=1)
    elif hh1 < hh2:
        diff = hh2 - hh1
        diff_top = int(diff/2)
        diff_bottom = hh2 - (diff - diff_top)
        inst2 = inst2[:,diff_top: diff_bottom,:]
        
    vc_dim = (inst1.shape[1], inst1.shape[2])
    dis_cnt = 0
    where_f = np.where(inst2==1)
    
    for ii in range(len(where_f[0])):
        nn1 = where_f[0][ii]
        nn2 = where_f[1][ii]
        nn3 = where_f[2][ii]
        if deform:
            ww_min = max(0,nn2-1)
            ww_max = min(vc_dim[0],nn2+1)
            hh_min = max(0,nn3-1)
            hh_max = min(vc_dim[1],nn3+1)
            
            if inst1[nn1, ww_min:ww_max+1, hh_min:hh_max+1].sum()==0:
                dis_cnt += 1
        else:
            if inst1[nn1,nn2,nn3]==0:
                dis_cnt += 1
    
    return (len(where_f[0]), dis_cnt)

def comp_one_to_ls(inst1, ls2, deform):
    inst_num2 = len(ls2)
    rst = np.zeros(inst_num2)
    for inst_nn2 in range(inst_num2):
        inst2 = ls2[inst_nn2]
        n1, n2 = vc_dis(inst1,inst2, deform)
        rst1 = n2/n1
        n1, n2 = vc_dis(inst2,inst1, deform)
        rst2 = n2/n1
        rst[inst_nn2] = min(rst1, rst2)
        
    return rst


def comp_two_ls(ls1, ls2, deform):
    inst_num1 = len(ls1)
    inst_num2 = len(ls2)
    mat = np.zeros((inst_num1,inst_num2))
    for inst_nn1 in range(inst_num1):
        mat[inst_nn1, :] = comp_one_to_ls(ls1[inst_nn1], ls2, deform)
            
    return mat

def vc_dis_sym(inst1, inst2, deform):
    n1,n2 = vc_dis(inst1, inst2, deform)
    rst1 = n2/n1
    n1,n2 = vc_dis(inst2, inst1, deform)
    rst2 = n2/n1
    return min(rst1, rst2)
    
def vc_dis_sym2(inpar):
    inst1, inst2 = inpar
    n1,n2,n3 = vc_dis_both(inst1, inst2)
    if n1==0:
        rst1_deform = 0.0
        rst1_nodeform = 0.0
    else:
        rst1_deform = n2/n1
        rst1_nodeform = n3/n1
    
    n1,n2,n3 = vc_dis_both(inst2, inst1)
    if n1==0:
        rst2_deform = 0.0
        rst2_nodeform = 0.0
    else: 
        rst2_deform = n2/n1
        rst2_nodeform = n3/n1
    
    return (np.mean([rst1_deform, rst2_deform]), np.mean([rst1_nodeform, rst2_nodeform]))

def vc_dis_paral(inpar):
    inst_ls, idx = inpar
    rst1 = np.ones(len(inst_ls))
    rst2 = np.ones(len(inst_ls))
    for nn in range(idx+1, len(inst_ls)):
        rst1[nn], rst2[nn] = vc_dis_sym2((inst_ls[idx], inst_ls[nn]))
        
    return (rst1, rst2)


def vc_dis_paral_full(inpar):
    inst_ls, inst2 = inpar
    rst1 = np.ones(len(inst_ls))
    rst2 = np.ones(len(inst_ls))
    for nn in range(len(inst_ls)):
        rst1[nn], rst2[nn] = vc_dis_sym2((inst_ls[nn], inst2))
        
    return (rst1, rst2)

    
def vc_dis_both(inst1, inst2):
    # inst1 = inst1[:,:,0:14]
    # inst2 = inst2[:,:,0:14]
    
    ww1 = inst1.shape[2]
    ww2 = inst2.shape[2]
    if ww1 > ww2:
        diff = ww1 - ww2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.concatenate([np.zeros((inst2.shape[0], inst2.shape[1], diff_top)), inst2], axis=2)
        inst2 = np.concatenate([inst2, np.zeros((inst2.shape[0], inst2.shape[1], diff_bottom))], axis=2)
    elif ww1 < ww2:
        diff = ww2 - ww1
        diff_top = int(diff/2)
        diff_bottom = ww2 - (diff - diff_top)
        inst2 = inst2[:, :, diff_top: diff_bottom]
        
    
    hh1 = inst1.shape[1]
    hh2 = inst2.shape[1]
    if hh1 > hh2:
        diff = hh1 - hh2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.concatenate([np.zeros((inst2.shape[0], diff_top, inst2.shape[2])), inst2], axis=1)
        inst2 = np.concatenate([inst2, np.zeros((inst2.shape[0], diff_bottom, inst2.shape[2]))], axis=1)
    elif hh1 < hh2:
        diff = hh2 - hh1
        diff_top = int(diff/2)
        diff_bottom = hh2 - (diff - diff_top)
        inst2 = inst2[:,diff_top: diff_bottom,:]
        
    vc_dim = (inst1.shape[1], inst1.shape[2])
    dis_cnt_deform = 0
    dis_cnt_nodeform = 0
    where_f = np.where(inst2==1)
    for ii in range(len(where_f[0])):
        nn1 = where_f[0][ii]
        nn2 = where_f[1][ii]
        nn3 = where_f[2][ii]
        
        ww_min = max(0,nn2-1)
        ww_max = min(vc_dim[0],nn2+1)
        hh_min = max(0,nn3-1)
        hh_max = min(vc_dim[1],nn3+1)
        
        if inst1[nn1, ww_min:ww_max+1, hh_min:hh_max+1].sum()==0:
            dis_cnt_deform += 1
        
        if inst1[nn1,nn2,nn3]==0:
            dis_cnt_nodeform += 1
            
    return (len(where_f[0]), dis_cnt_deform, dis_cnt_nodeform)
 

def vc_dist_rigid_transfer_func(inst1, inst2, shft1, shft2):
    assert(inst1.shape[0]==inst2.shape[0]) # row num
    assert(inst1.shape[1]==inst2.shape[1]) # col num
    assert(inst1.shape[2]==inst2.shape[2]) # vc num
    dim1, dim2 = inst1.shape[0:2]
    if shft1 == -1:
        inst1_s = inst1[0:dim1-1, :, :]
        inst2_s = inst2[1:, :, :]
    elif shft1 == 0:
        inst1_s = inst1
        inst2_s = inst2
    elif shft1 == 1:
        inst1_s = inst1[1:, :, :]
        inst2_s = inst2[0:dim1-1, :, :]
    
    '''
    try:
        assert(inst1_s.shape[0]==inst2_s.shape[0]) # row num
        assert(inst1_s.shape[1]==inst2_s.shape[1]) # col num
        assert(inst1_s.shape[2]==inst2_s.shape[2]) # vc num
    except:
        print(inst1.shape, inst2.shape, inst1_s.shape, inst2_s.shape, shft1, shft2)
    '''
    
    
    if shft2 == -1:
        inst1_s = inst1_s[:, 0:dim2-1, :]
        inst2_s = inst2_s[:, 1:, :]
    elif shft2 == 1:
        inst1_s = inst1_s[:, 1:, :]
        inst2_s = inst2_s[:, 0:dim2-1, :]
    
    '''
    try:
        assert(inst1_s.shape[0]==inst2_s.shape[0]) # row num
        assert(inst1_s.shape[1]==inst2_s.shape[1]) # col num
        assert(inst1_s.shape[2]==inst2_s.shape[2]) # vc num
    except:
        print(inst1_s.shape, inst2_s.shape, shft1, shft2)
    '''    
        
    ovlp = np.sum(np.logical_and(inst1_s, inst2_s))
    return np.mean([(np.sum(inst1)-ovlp)/np.sum(inst1), (np.sum(inst2)-ovlp)/np.sum(inst2)])
    
    
def vc_dist_rigid_transfer_sym(inpar):
    inst1, inst2 = inpar
    rst = []
    for shft1 in [-1,0,1]:
        for shft2 in [-1,0,1]:
            rst.append(vc_dist_rigid_transfer_func(inst1, inst2, shft1, shft2))
            
    return np.min(rst)


def kernel_kmeans(Kern, K, finit = None):
    print('start kernel kmeans')
    N = Kern.shape[0]
    A = np.zeros((N, K))
    if finit is None:
        f = np.random.choice(K, N).astype(int)
    else:
        f = finit
        
    for nn in range(N):
        A[nn,f[nn]] = 1
        
    change = 1
    for itt in range(5000):
        change = 0
        E = A.dot(np.diag(1./np.sum(A, axis=0)))
        Z = np.ones((N, 1)) * np.diag(E.T.dot(Kern.dot(E))) -2*Kern.dot(E)
        ff = np.argmin(Z, axis=1)
        for nn in range(N):
            if f[nn] != ff[nn]:
                change=1
                A[nn, f[nn]] = 0
                A[nn, ff[nn]] = 1
                
        cc = sum(np.min(Z, axis=1))
        print('iter {0}: {1}'.format(itt, cc))
            
        if np.isnan(cc):
            break
            
        if change == 0:
            break
            
        f = ff
    
    return f, sum(np.min(Z, axis=1))


def psd_proj(W):
    W2 = 0.5*(W+W.T)
    w1,v1 = np.linalg.eig(W2)
    w1[w1<0] = 0
    W3 = v1.dot(np.diag(w1).dot(v1.T))
    return W3

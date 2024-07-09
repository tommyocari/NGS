import numpy as np
import pandas as pd
import torch

def binning_equal_q(sorted_log10p_vector, inv_sorted_log10q_vector, counts, bins=150, writefolder=False, step=100):
    
    n_q_per_bin = len(inv_sorted_log10q_vector) // (bins)

    df_bins = pd.DataFrame(columns=['edge_sx','edge_dx','n_lambdas','n_lambdas_inferred_aritm','n_lambdas_inferred_geo','n_nonzeros','mean_aritm_lambda','mean_geom_lambda','mean_lambda','var_uniform_lambda','var_lambda','mean_count','var_count'])
    logR = np.log10(counts.sum())
    
    for i in range(bins):
        print(i)
        if i != bins-1:
            counts_sel = counts[i*n_q_per_bin:(i+1)*n_q_per_bin]
            edge_sx = inv_sorted_log10q_vector[i*n_q_per_bin]
            edge_dx = inv_sorted_log10q_vector[(i+1)*n_q_per_bin]
        else:
            counts_sel = counts[i*n_q_per_bin:-1]
            edge_sx = inv_sorted_log10q_vector[i*n_q_per_bin]
            edge_dx = inv_sorted_log10q_vector[-1]
        
        lambda_sx = 10**(edge_sx + logR)
        lambda_dx = 10**(edge_dx + logR)
        index_sx = torch.searchsorted(sorted_log10p_vector, edge_sx, side='right')
        index_dx = torch.searchsorted(sorted_log10p_vector, edge_dx, side='right')
        print(index_sx, index_dx)
        logp_sel = sorted_log10p_vector[index_dx:index_sx] 
        
        n_lambdas = len(logp_sel)
        print('elements in the bin: %d'%(n_lambdas))
        n_nonzeros = len(counts_sel)
        print('nonzeros: %d'%(n_nonzeros))
        mean_aritm_lambda = 0.5*(lambda_sx + lambda_dx)
        mean_geom_lambda = np.sqrt(lambda_sx * lambda_dx)
        mean_lambda = (10**(logp_sel + logR)).mean().numpy()
        var_uniform_lambda = (lambda_dx - lambda_sx)**2 / 12
        var_lambda = (10**(logp_sel + logR)).var().numpy()
        mean_count = counts_sel.sum() / n_lambdas
        var_count = (counts_sel**2).sum() / n_lambdas - mean_count**2
        n_lambdas_inferred_aritm = counts_sel.sum() / mean_aritm_lambda
        n_lambdas_inferred_geo = counts_sel.sum() / mean_geom_lambda
        df_bins.loc[i] = [edge_sx,edge_dx,n_lambdas,n_lambdas_inferred_aritm,n_lambdas_inferred_geo,n_nonzeros,mean_aritm_lambda,mean_geom_lambda,mean_lambda,var_uniform_lambda,var_lambda,mean_count,var_count]
        
        if writefolder and i%step==0:
            count, frequencies = np.unique(counts_sel, return_counts=True)
            count = np.insert(count, 0, 0)
            np.save(writefolder+'/counts_%d.npy'%i,count)
            frequencies = np.insert(frequencies, 0, n_lambdas-n_nonzeros)
            np.save(writefolder+'/frequencies_%d.npy'%i,frequencies)
            
    return df_bins

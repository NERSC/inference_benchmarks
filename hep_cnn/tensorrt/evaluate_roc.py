from sklearn import metrics
import numpy as np
import h5py as h5
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--input_path",type=str,help="Path to input hdf5-files.")
    AP.add_argument("--output_path",type=str,help="Path to store ROC curves.")
    parsed = AP.parse_args()

    #get all files
    h5files = [os.path.join(parsed.input_path, x) for x in os.listdir(parsed.input_path) if x.endswith(".h5")]
    
    #linestyles
    lslist = ['-', ':']
    
    #compute the roc on the intersection
    results=[]
    for idx, filename in enumerate(h5files):
            
        #read predictions labels, weight and psr
        with h5.File(filename,'r') as f:
            predictions = f['prediction'][...]
            labels = f['label'][...]
            psr = f['psr'][...]
            weights = f['weight'][...]
        
        #do the roc:
        fpr, tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1, sample_weight=weights)
        fpr_cut, tpr_cut, _ = metrics.roc_curve(labels, psr, pos_label=1, sample_weight=weights)
        
        #add to results dict
        results.append({'name': filename, 'fpr': fpr, 'tpr': tpr, 'fprc': fpr_cut, 'tprc': tpr_cut})
    
    #plot the data
    plt.figure()
    lw = 2
    #full curve
    for idx, item in enumerate(results):
        plt.plot(item['fpr'], item['tpr'],
                 lw=lw, linestyle=lslist[idx], label='ROC curve {} (area = {:0.2f})'.format(item['name'], metrics.auc(item['fpr'],item['tpr'],reorder=True)))
    
    plt.scatter([results[0]['fprc']],[results[0]['tprc']], label='standard cuts')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(parsed.output_path,'ROC_1400_850.png'),dpi=300)

    #zoomed-in
    plt.xlim([0.0, 0.0004])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(parsed.output_path,'ROC_1400_850_zoom.png'),dpi=300)

if __name__ == "__main__":
    main()

import numpy as np
from scipy import stats
import h5py as h5
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab


def get_hist_bins(data, bins=None, get_error=False, range=(-1.1,1.1)):
    bins = 60 if bins is None else bins
    y, x = np.histogram(data, bins=bins, range=range)
    #normalize y by number of entries
    #y = y / float(data.shape[0])
    #merge overflow bin
    x = 0.5*(x[1:]+x[:-1])
    if get_error == True:
        y_err = np.sqrt(y) / float(data.shape[0])
        y = y / float(data.shape[0])
        return x, y, y_err
    else:
        return x, y


def plot_pixel_histograms(fakelist, test, fakefilenames, output_path):
  
  #create lists for outputs
  #real:
  test_bins, test_hist, test_err = get_hist_bins(test, get_error=True)
  
  #fake
  fake_bins_list=[]
  fake_hist_list=[]
  fake_err_list=[]
  ks_list=[]
  for fake in fakelist:
    tmp_fake_bins, tmp_fake_hist, tmp_fake_err = get_hist_bins(fake, get_error=True)
    fake_bins_list.append(tmp_fake_bins)
    fake_hist_list.append(tmp_fake_hist)
    fake_err_list.append(tmp_fake_err)
    ks_list.append(stats.ks_2samp(test_hist, tmp_fake_hist)[1])
  
  fig, ax = plt.subplots(figsize=(7,6))
  #plot test
  ax.errorbar(test_bins, test_hist, yerr=test_err, fmt='--ks', label='Simulated', markersize=7)

  # plot generated
  for i in range(len(fake_bins_list)):
    fake_label = 'DC-GAN (KS=%2.3f) '%ks_list[i]+fakefilenames[i]
    ax.errorbar(fake_bins_list[i], fake_hist_list[i], yerr=fake_err_list[i], fmt='o', \
               label=fake_label, linewidth=2, markersize=6);

  ax.legend(loc="best", fontsize=10)
  ax.set_yscale('log');
  ax.set_xlabel('Pixel Intensity', fontsize=18);
  ax.set_ylabel('Counts (arb. units)', fontsize=18);
  plt.tick_params(axis='both', labelsize=15, length=5)
  plt.tick_params(axis='both', which='minor', length=3)
  #plt.ylim(5e-10, 8*10**7)
  plt.xlim(-0.3,1.1)
  plt.title('Pixels distribution', fontsize=16);

  plt.savefig(os.path.join(output_path,'pixel_intensity.pdf'), bbox_inches='tight', format='pdf')
  plt.close('all')


def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--input_path_real",type=str,help="Path to real data.")
    AP.add_argument("--input_path_fake",type=str,help="Path to fake data.")
    AP.add_argument("--output_path",type=str,help="Path to store ROC curves.")
    parsed = AP.parse_args()
    
    #parse path
    realfiles = [os.path.join(parsed.input_path_real, x) for x in os.listdir(parsed.input_path_real) if x.endswith('.h5')]
    fakefiles = [os.path.join(parsed.input_path_fake, x) for x in os.listdir(parsed.input_path_fake) if x.endswith('.h5')]
    
    #load data
    #real data
    realdatalist = []
    for realfile in realfiles:
        with h5.File(realfile, 'r') as f:
            data = f['data'][...]
            realdatalist.append(data)
    
    #fake data
    fakedatalist = []
    for fakefile in fakefiles:
        with h5.File(fakefile, 'r') as f:
            data = f['data'][...]
            fakedatalist.append(data)
    
    #plot hists
    #merge the real files
    realdata = np.vstack(realdatalist)
    
    #plot the histograms
    plot_pixel_histograms(fakedatalist, realdata, fakefiles, parsed.output_path)
    
if __name__ == "__main__":
    main()



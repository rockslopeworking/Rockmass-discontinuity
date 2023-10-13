import numpy as np
import hdbscan
import tkinter as tk
from tkinter import filedialog
import time
root = tk.Tk()
root.withdraw()
#HDBSCAN copyright{L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017}
file_path = filedialog.askopenfilename()#Select J_pcData saved in section8.1 in the file box
print('The selected file path is：{}'.format(file_path))
#
import scipy.io
A = scipy.io.loadmat(file_path)
print(A.keys())
J_pcData=A['J_pcData']
print(J_pcData)
K_C=J_pcData.shape[0]
print(K_C)
#
file_pathsave=filedialog.askdirectory()#In which folder to save the HDBSCAN results.
start=time.time()
for ii in range(K_C):
    J=J_pcData[ii][0][:,0:3]
    print(J)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10,gen_min_span_tree=True)#min_samples=5
    cluster_labels1 = clusterer.fit_predict(J)
    print(cluster_labels1)
    a=cluster_labels1.shape
    print(a)
    jj=str(ii+1)
    np.savetxt(file_pathsave+'/cluster_labels'+jj+'.txt',cluster_labels1)
end=time.time()
run_time = end - start    #
print(run_time)

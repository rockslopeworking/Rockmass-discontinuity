import numpy as np
import hdbscan
import tkinter as tk
from tkinter import filedialog
import time
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()#在文件框选择section8.1 保存的 J_pcData
print('选择的文件路径为：{}'.format(file_path))
#
import scipy.io
A = scipy.io.loadmat(file_path)
print(A.keys())
J_pcData=A['J_pcData']
print(J_pcData)
K_C=J_pcData.shape[0]
print(K_C)
#
file_pathsave=filedialog.askdirectory()#在哪个文件夹保存聚类HDBSCAN结果。
start=time.time()
for ii in range(K_C):
    J=J_pcData[ii][0][:,0:3]
    print(J)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10,min_samples=60,gen_min_span_tree=True)#min_samples=5
    cluster_labels1 = clusterer.fit_predict(J)
    print(cluster_labels1)
    a=cluster_labels1.shape
    print(a)
    import csv
    jj=str(ii+1)
    np.savetxt(file_pathsave+'/cluster_labels'+jj+'.txt',cluster_labels1)
end=time.time()
run_time = end - start    # 程序的运行时间，单位为秒
print(run_time)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
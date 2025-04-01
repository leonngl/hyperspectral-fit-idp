import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

path = "/home/caredda/DVP/simulation/CREATIS-UCL-White-Monte-Carlo-Framework/spectra/cyt_spectra_in_UCL_repo/"
out_path = path+"out/"

#*1000 to convert in cm.Mol-1
F = 1000
oxCytaa3 = np.loadtxt(path+"oxCytaa3.txt")
oxCytb = np.loadtxt(path+"oxCytb.txt")
oxCytc = np.loadtxt(path+"oxCytc.txt")
redCytaa3 = np.loadtxt(path+"redCytaa3.txt")
redCytb = np.loadtxt(path+"redCytb.txt")
redCytc = np.loadtxt(path+"redCytc.txt")
eps_Hb = np.loadtxt(path+"eps_Hb.txt")
eps_HbO2 = np.loadtxt(path+"eps_HbO2.txt")
mua_Fat = np.loadtxt(path+"mua_Fat.txt")
mua_H2O = np.loadtxt(path+"mua_H2O.txt")
w_Hb_mua = np.loadtxt(path+"lambda.txt")


#CCO
w_Cytaa3 = oxCytaa3[:,0]
w_add = np.arange(w_Hb_mua[0],w_Cytaa3[0])
w_Cytaa3 = np.insert(w_Cytaa3,0,w_add,axis=0)
oxCytaa3 = np.insert(oxCytaa3[:,1],0,np.zeros(w_add.shape),axis=0)
redCytaa3 = np.insert(redCytaa3[:,1],0,np.zeros(w_add.shape),axis=0)


oxCytaa3 = interpolate.interp1d(w_Cytaa3,oxCytaa3, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F
redCytaa3 = interpolate.interp1d(w_Cytaa3,redCytaa3, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F


#Cyt C
w_Cytc = oxCytc[:,0]
w_add = np.arange(w_Hb_mua[0],w_Cytc[0])
w_Cytc = np.insert(w_Cytc,0,w_add,axis=0)
oxCytc = np.insert(oxCytc[:,1],0,np.zeros(w_add.shape),axis=0)
redCytc = np.insert(redCytc[:,1],0,np.zeros(w_add.shape),axis=0)


oxCytc = interpolate.interp1d(w_Cytc,oxCytc, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F
redCytc = interpolate.interp1d(w_Cytc,redCytc, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F


#Cyt b
w_Cytb = oxCytb[:,0]
w_add = np.arange(w_Hb_mua[0],w_Cytb[0])
w_Cytb = np.insert(w_Cytb,0,w_add,axis=0)
oxCytb = np.insert(oxCytb[:,1],0,np.zeros(w_add.shape),axis=0)
redCytb = np.insert(redCytb[:,1],0,np.zeros(w_add.shape),axis=0)


oxCytb = interpolate.interp1d(w_Cytb,oxCytb, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F
redCytb = interpolate.interp1d(w_Cytb,redCytb, kind='cubic',fill_value="extrapolate")(w_Hb_mua)*F


##
np.savetxt(out_path+"eps_HbO2.txt",eps_HbO2,newline=' ')
np.savetxt(out_path+"eps_Hb.txt",eps_Hb,newline=' ')
np.savetxt(out_path+"mua_Fat.txt",mua_Fat,newline=' ')
np.savetxt(out_path+"mua_H2O.txt",mua_H2O,newline=' ')
np.savetxt(out_path+"lambda.txt",w_Hb_mua,newline=' ')
np.savetxt(out_path+"eps_oxCCO.txt",oxCytaa3,newline=' ')
np.savetxt(out_path+"eps_redCCO.txt",redCytaa3,newline=' ')
np.savetxt(out_path+"eps_oxCytc.txt",oxCytc,newline=' ')
np.savetxt(out_path+"eps_redCytc.txt",redCytc,newline=' ')
np.savetxt(out_path+"eps_oxCytb.txt",oxCytb,newline=' ')
np.savetxt(out_path+"eps_redCytb.txt",redCytb,newline=' ')


## plot

ft_title = 14
lw = 3

plt.close('all')
plt.figure()
plt.suptitle("Extinction molar coefficients",fontsize = ft_title)

plt.subplot(121)
plt.plot(w_Hb_mua,eps_HbO2,'r',linewidth = lw, label="$\epsilon_{HbO_2}$")
plt.plot(w_Hb_mua,eps_Hb,'b',linewidth = lw, label="$\epsilon_{Hb}$")
plt.plot(w_Hb_mua,oxCytaa3,'g',linewidth = lw, label="$\epsilon_{oxCCO}$")
plt.plot(w_Hb_mua,redCytaa3,'g:',linewidth = lw, label="$\epsilon_{redCCO}$")

plt.plot(w_Hb_mua,oxCytb,'m',linewidth = lw, label="$\epsilon_{oxCytb}$")
plt.plot(w_Hb_mua,redCytb,'m:',linewidth = lw, label="$\epsilon_{redCytb}$")
plt.plot(w_Hb_mua,oxCytc,'k',linewidth = lw, label="$\epsilon_{oxCytc}$")
plt.plot(w_Hb_mua,redCytc,'k:',linewidth = lw, label="$\epsilon_{redCytc}$")
plt.yscale("symlog")
plt.xlabel("Wavelength (nm)",fontsize = ft_title)
plt.ylabel("Molar extinction coefficient ($mol^{-1}.L.cm^{-1}$",fontsize = ft_title)
plt.grid()
plt.legend(loc="best",fontsize = ft_title)
plt.xlim(w_Hb_mua[0],w_Hb_mua[-1])


plt.xticks(fontsize=ft_title)

plt.subplot(122)
plt.plot(w_Hb_mua,eps_HbO2,'r',linewidth = lw, label="$\epsilon_{HbO_2}$")
plt.plot(w_Hb_mua,eps_Hb,'b',linewidth = lw, label="$\epsilon_{Hb}$")
plt.plot(w_Hb_mua,oxCytaa3-redCytaa3,'g',linewidth = lw, label="$\epsilon_{diffCCO}$")
plt.plot(w_Hb_mua,oxCytb-redCytb,'m',linewidth = lw, label="$\epsilon_{diffCytb}$")
plt.plot(w_Hb_mua,oxCytc-redCytc,'k',linewidth = lw, label="$\epsilon_{diffCytc}$")

plt.yscale("symlog")
plt.xlabel("Wavelength (nm)",fontsize = ft_title)
plt.ylabel("Molar extinction coefficient ($mol^{-1}.L.cm^{-1}$",fontsize = ft_title)
plt.grid()
plt.legend(loc="best",fontsize = ft_title)
plt.xlim(w_Hb_mua[0],w_Hb_mua[-1])


plt.xticks(fontsize=ft_title)
plt.show()


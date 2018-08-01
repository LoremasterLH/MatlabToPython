# 1-Vzorčenje
# Predlagan IDE PyCharm Community Edition (na voljo za Windows, macOS in Linux)
# https://www.jetbrains.com/pycharm/download
# Navodila za pridobitev potrebnih knjižnic:
# https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html
import numpy as np
import scipy.signal
from matplotlib import cm  # color mapping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sounddevice as sd
import plotly.plotly as py
import plotly.graph_objs as go
import os
from PIL import Image

# ----------------------------------------------------------------------------------------
# Vzorčenje in Nyquistov teorem;
Fvz = 100  # Frekvenca vzorčenja (v Hz)
T = 1  # dolžina signala (v s)
i = np.arange(float(T)*Fvz)/Fvz  # vektor časovnih indeksov
f1 = 5  # frekvenca sinusoide
A1 = 1  # amplituda sinusoide
faza1 = 0.0  # faza sinusoide

# ukazi.m:10
# izris sinuside pri razlicnih fazah......................................................
plt.figure()
for faza1 in np.arange(0, 6.1, 0.1).reshape(-1):
    plt.cla()
    # pri množenju z matriko je potrebno uporabiti numpy.np.dot(...)
    s = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))
    plt.plot(i, s)
    plt.axis('tight')
    setattr(plt.gca, 'YLim', [-1, 1])
    plt.title('Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda (dB)')
    plt.waitforbuttonpress()

# ukazi.m:23
# izris sinusid pri razlicnih frekvencah...............................................
faza1 = 0.0
plt.figure()
for f1 in np.arange(Fvz+1).reshape(-1):
    plt.cla()
    s = np.dot(A1, np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi))
    plt.plot(i, s)
    plt.autoscale(False)
    plt.ylim(-1,1)
    plt.title('Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda (dB)')
    plt.pause(0.05)

# ukazi.m:37
# ########################### izris sinusoid s frekvenco f1 in Fvz-f1.........................
plt.figure()
f1 = 1
s1 = np.sin(np.dot(2*np.pi*f1, i) + faza1*np.pi)
plt.plot(i, s1, 'b')
f2 = Fvz - f1
s2 = np.sin(np.dot(np.dot(np.dot(2, np.pi), f2), i) + np.dot(faza1, np.pi))
plt.plot(i, s2, 'r')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.title('Fvz = {0} Hz, Frekvenca1 = {1} Hz, Frekvenca2 = {2} Hz, faza = {3} $\pi$'.format(Fvz, f1, f2, faza1))
plt.axis('tight')
# ----------------------------------------------------------------------------------------

# glej tudi primera v Mathematici: (WheelIllusion.nbp in SamplingTheorem.nbp)

# V Z O R Č E N J E    Z V O K A
# -----------------------------------------------------------------------------------------
# ukazi.m:56
# vzorenje zvoka
Fs = 44100  # vzorčevalna frekvenca
bits = 16  # bitna ločljivost
nchans = 1  # 1 (mono), 2 (stereo).
posnetek = sd.rec(5*Fs, Fs, nchans, blocking=True)
plt.figure()
plt.plot(posnetek)

sd.play(posnetek, 44100)
sd.play(posnetek, 44100 / 2)
sd.play(posnetek, 2 * 44100)

# -----------------------------------------------------------------------------------------
# ukazi.m:73
# ali zaznate fazne spremembe? Spreminjajte faza1 med 0 in 2.0 in poženite ta demo...
Fvz = 44100  # vzorčevalna frekvenca
T = 3  # čas v sekundah
i = np.arange(0.0, T*Fvz+1, 1)/Fvz  # vektor časovnih indeksov
f1 = 500  # frekvenca sinusoide
A1 = 0.3  # amplituda sinusoide
faza1 = 1.0  # faza sinusoide
s = np.dot(A1, np.sin(np.dot(2*np.pi*f1, i) + faza1*np.pi))
s2 = np.dot(A1, np.sin(np.dot(2*np.pi*f2, i) + 0*np.pi))
sd.play(np.concatenate((s, s2)), Fvz)  # dvojni oklepaji pri concatenate, ker sta oba prvi parameter
# "navadna" polja združiš z s+s2, v numpy.array pa to sešteje istoležne elemente


# -----------------------------------------------------------------------------------------
# ukazi.m:88
# trije poskusi: 1. f1 = 50;
#                2. f1 = 450;
#                3. f1 = 1450;
#                4. f1 = 2450;

Fvz = 44100  # vzorčevalna frekvenca
T = 3  # čas v sekundah
i = np.arange(T*Fvz+1)/Fvz  # vektor časovnih indeksov
f1 = 50  # frekvenca sinusoide
A1 = 5.5  # amplituda sinusoide
faza1 = 0.0  # faza sinusoide
f2 = f1 + 1  # frekvenca druge sinusoide

s1 = np.dot(A1, np.sin(np.dot(2*np.pi*f1, i) + faza1*np.pi))  # tvorjenje prve sinusoide
s2 = np.dot(A1, np.sin(np.dot(2*np.pi*f2, i) + faza1*np.pi))  # tvorjenje druge sinusoide
sd.play(np.concatenate((s, s2)), Fvz)

# -----------------------------------------------------------------------------------------
# ukazi.m:106
# ali zaznate zvok netopirja pri 90000 Hz? Nyquist?
Fvz = 44100
T = 3
i = np.arange(T*Fvz+1)/Fvz  # vektor časovnih indeksov
fnetopir = 140000  # frekvenca sinusoide
A1 = 5.5  # amplituda sinusoide
faza1 = 1.0  # faza sinusoide

s = np.dot(A1, np.sin(np.dot(2*np.pi*fnetopir, i) + faza1*np.pi))  # tvorjenje sinusoide
sd.play(s, Fvz)

# ukazi.m:118
# izris sinuside pri razlicnih fazah (verzija 2)......................................................
Fvz = 100
T = 1
i = np.arange(T*Fvz+1)/Fvz
f1 = 5
A1 = 5
faza1 = 0.0

# spremninjanje frekvence...
plt.close('all')
# horribly slow
plt.figure()
for f1 in np.arange(0, Fvz+1).reshape(-1):
    plt.clf()
    s = np.sin(np.dot(2 * np.pi * f1, i) + faza1 * np.pi)
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.axis('tight')
    setattr(plt.gca, 'YLim', [-1, 1])
    plt.title('Časovna domena: Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.subplot(2, 1, 2)

    plt.plot(abs(np.fft.fft(s)), 'r')
    plt.axis('tight')
    setattr(plt.gca, 'YLim', [-1, 1])
    plt.title('Frekvenčna domena (abs): Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.waitforbuttonpress()


# in faze... (več o tem na naslednjih vajah)
f1 = 5
A1 = 5
faza1 = 0.0
plt.close('all')
plt.figure()
for faza1 in np.arange(0, 2, 0.1).reshape(-1):
    plt.clf()
    s = np.sin(np.dot(2*np.pi*f1, i) + faza1*np.pi)
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.axis('tight')
    setattr(plt.gca, 'YLim', [- 1, 1])
    plt.title('Časovna domena: Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.subplot(2, 1, 2)
    plt.plot(abs(np.fft.fft(s)), 'r')
    plt.axis('tight')
    setattr(plt.gca, 'YLim', [- 1, 1])
    plt.title('Frekvenčna domena (abs): Fvz = {0} Hz, Frekvenca = {1} Hz, faza = {2} $\pi$'.format(Fvz, f1, faza1))
    plt.pause(0.05)

# S L I K E
# -----------------------------------------------------------------------------------------
# ukazi.m:181
# vzorenje slik in Moire

A = plt.imread(os.path.join(os.path.curdir, 'Moire.jpg'))
plt.figure()
plt.axis('off')
plt.imshow(A)
plt.title('originalna slika')

pvz = 3
plt.figure()
plt.axis('off')
plt.imshow(A[::pvz, ::pvz])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))
st_bit = 0

kvant = 2 ** (9 - st_bit)
plt.figure()
plt.imshow(np.dot((A[:, :, :] / kvant), kvant))
plt.title('slika pri bitni ločljivosti {0}'.format(st_bit))
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.dot((A[:, :, :] / kvant), kvant))
plt.title('slika pri bitni ločljivosti {0}'.format(st_bit))
plt.subplot(2, 2, 2)
plt.imshow(np.dot((A[:, :, 1] / kvant), kvant))
plt.title('ravnina R pri bitni ločljivosti {0}'.format(st_bit))
plt.subplot(2, 2, 3)
plt.imshow(np.dot((A[:, :, 2] / kvant), kvant))
plt.title('ravnina G pri bitni ločljivosti {0}'.format(st_bit))
plt.subplot(2, 2, 4)
plt.imshow(np.dot((A[:, :, 3] / kvant), kvant))
plt.title('ravnina B pri bitni ločljivosti {0}'.format(st_bit))

# -----------------------------------------------------------------------------------------
# ukazi.m:215
# spekter slik, Moire in Diskretna Fourierova transformacija (fft2)
A = plt.imread(os.path.join(os.path.curdir, 'Moire.jpg'))
plt.figure()
plt.imshow(A)
plt.title('originalna slika')
plt.close('all')

B = np.double(A[:, :, 1])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('R ravnina')
plt.show()

# ukazi.m:226
B = np.double(A[:, :, 2])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('G ravnina')
plt.show()

# ukazi.m:228
B = np.double(A[:, :, 3])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('B ravnina')
plt.show()

# ukazi.m:231
# prevzorčena slika..................................
pvz = 4
B = np.double(A[0::pvz, 0::pvz, 1])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('R ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# ukazi.m:235
B = np.double(A[0::pvz, 0::pvz, 2])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('G ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# ukazi.m:237
B = np.double(A[0::pvz, 0::pvz, 3])
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Z = abs(np.fft.fft2(B - np.mean(np.ravel(B)))[0:100, 0:100])
colors = cm.Blues(Z)
rcount, ccount, _ = colors.shape

fig = plt.figure()
fig.set_size_inches(7, 7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))
plt.title('B ravnina, po podvzorenju s faktorjem {0}'.format(pvz))
plt.show()

# -----------------------------------------------------------------------------------------
# ukazi.m:241
# podvzorenje slik in operator povpreenja

A = plt.imread(os.path.join(os.path.curdir, 'Moire.jpg'))
plt.figure()
plt.imshow(A)
plt.title('originalna slika')

pvz = 3  # faktor podvzorčenja
plt.figure()
plt.imshow(A[0::pvz, 0::pvz, :])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))

# ukazi.m:253
# operator povprečenja (verzija 1)
D = 3  # premer lokalne okolice piksla, na kateri se izračuna povprečna verdnost
B = np.ndarray((A.shape[0] - D+1, A.shape[1] - D+1, D))  # np.matrix?
for r in np.arange(0, A.shape[0] - D).reshape(-1):
    for c in np.arange(0, A.shape[1] - D).reshape(-1):
        C = A[r + np.arange(0, D-1), c + np.arange(0, D-1), 0]
        B[r, c, 0] = np.mean(np.ravel(C))
        C = A[r + np.arange(0, D-1), c + np.arange(0, D-1), 1]
        B[r, c, 1] = np.mean(np.ravel(C))
        C = A[r + np.arange(0, D-1), c + np.arange(0, D-1), 2]
        B[r, c, 2] = np.mean(np.ravel(C))

plt.figure()
plt.imshow(np.uint8(B))
plt.title('zglajena slika')
# operator povpreenja (verzija 2)
# isti operator povpreenja kot zgoraj, implementiran nekoliko drugae (veliko hitreja izvedba)
D = 3
# ukazi.m:270

B = np.ndarray((A.shape[0]+D-1, A.shape[1]+D-1, D))
# ukazi.m:271

B[:, :, 0] = scipy.signal.convolve2d(np.double(A[:, :, 0]), np.ones((D, D), np.float) / D ** 2)
# ukazi.m:272
B[:, :, 1] = scipy.signal.convolve2d(np.double(A[:, :, 1]), np.ones((D, D), np.float) / D ** 2)
# ukazi.m:273
B[:, :, 2] = scipy.signal.convolve2d(np.double(A[:, :, 2]), np.ones((D, D), np.float) / D ** 2)
# ukazi.m:274
B = np.uint8(B)
# ukazi.m:275

plt.figure()
plt.imshow(B)
plt.title('zglajena slika')
# prikaz
pvz = 3
# ukazi.m:281

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(A[0::pvz, 0::pvz, :])
plt.title('podvzorčena slika: faktor podvzorčenja {0}'.format(pvz))
plt.subplot(1, 2, 2)
plt.imshow(B[1::pvz, 1::pvz, :])
plt.title('zglajena podvzorena slika: faktor podvzorenja {0}'.format(pvz))

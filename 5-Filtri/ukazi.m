' DIGITALNI FILTRI'
'-------------------------------------------------------------------
% filtriranje v frekven�nem prostoru
close all;  clc;
Fsamp = 1024; % vzor�evalna frekvenca
LomnaFrek = 50; % lomna frekvenca - IGRAJTE SE Z VREDNOSTJO !!!

figure;
x=randn(1,1*Fsamp); % signal, ki ga �elimo filtrirati
X=fft(x);        % pretvorimo v frekven�ni prostor
F=[ones(1,LomnaFrek) zeros(1,length(x)-2*LomnaFrek+1) ones(1,LomnaFrek-1)]; % nizko sito
subplot(2,1,1); hold on;
plot(abs(X)/max(abs(X(:))));   % normalizirana FFT trasfromiranka signala X
plot(F,'r','LineWidth',2);     % frekven�na karakteristika filtra
xlabel('f (Hz)'); ylabel('amplituda');
title(['Lomna frekvenca = ' num2str(LomnaFrek) ' Hz'])
axis tight;

R=X.*F;          % mno�enje v frekven�nem prostoru
r=ifft(R);       % produkt pretvorimo nazaj
subplot(2,1,2);  hold on;
plot(x);         % izri�emo originalni signal
plot(r,'r','LineWidth',2);         % izri�emo rezultat filtriranja
xlabel('t (vzorci)'); ylabel('amplituda');
axis tight;

'-------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% impulzni odziv filtra - antivzro�nost, dol�ina filtra, konvolucija v �asovni domeni...
close all; clc;
Fsamp = 1024; % vzor�evalna frekvenca
LomnaFrek = 50; % - IGRAJTE SE Z VREDNOSTJO !!!
F=[ones(1,LomnaFrek) zeros(1,Fsamp-2*LomnaFrek+1) ones(1,LomnaFrek-1)]; % nizko sito

f = ifftshift(ifft(F)); % zaradi periodi�nosti inverzne Fourierove transformiranke lahko
% �asovno os prestravimo za length(F)/2 vzorcev v levo.

figure;
subplot(3,1,1);
plot([-length(f)/2:length(f)/2-1],f);
xlabel('t (vzorci)'); ylabel('amplituda');
axis tight;
title(['impulzni odziv frekven�nega filtra z Lomno frekvenco = ' num2str(LomnaFrek) ' Hz']);
subplot(3,1,2);
plot([-length(f)/2:length(f)/2-1],abs(f));
xlabel('t (vzorci)'); ylabel('amplituda');
title('absolutna vrednost impulznega odziva frekven�nega filtra')
%set(gca,'YScale','log');
axis tight;
subplot(3,1,3);
plot([-length(f)/2:length(f)/2-1],20*log10(abs(f)));
xlabel('t (vzorci)'); ylabel('amplituda (dB)');
title('absolutna vrednost impulznega odziva frekven�nega filtra')
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% poskusimo narediti filter kon�en in vzro�en
% impulzni odziv se �e nahaja v spr. impulzni
LomnaFrek = 200;
Fsamp = 1024; % vzor�evalna frekvenca
F=[ones(1,LomnaFrek) zeros(1,Fsamp-2*LomnaFrek+1) ones(1,LomnaFrek-1)]; % nizko sito
f = ifftshift(ifft(F)); % zaradi periodi�nosti inverzne Fourierove transformiranke lahko
% �asovno os prestravimo za length(F)/2 vzorcev v levo.
f2 = f;
f2(512:1024)=f2(512-20:1024-20); %pomik desno (naredimo filter vzro�en)
f2(1:511)=0; % Postavljanje amplitud antivzro�nega dela na 0 (naredimo filter vzro�en in kon�en)
f2(512+100:1024)=0; % pore�emo �e rep
F2=fft(f2);  % FFT transformiranka novega impulznega odziva

close all; clc;
figure;
subplot(3,1,1); hold on;
plot([-length(f2)/2:length(f2)/2-1],f2,'r');
plot([-length(f)/2:length(f)/2-1],f,'b');
xlabel('t (vzorci)'); ylabel('amplituda');
axis tight;
title(['impulzni odziv frekven�nega filtra z Lomno frekvenco = ' num2str(LomnaFrek) ' Hz']);
subplot(3,1,2); hold on;
plot([-length(f2)/2:length(f2)/2-1],20*log10(abs(f2)),'r');
plot([-length(f)/2:length(f)/2-1],20*log10(abs(f)));
xlabel('t (vzorci)'); ylabel('amplituda (dB)');
title('absolutna vrednost frekven�ne karakteristike filtra')
axis tight;
subplot(3,1,3);  hold on;
plot(abs(F));
plot(abs(F2),'r');
xlabel('f (Hz)'); ylabel('amplituda');
title('absolutna vrednost frekven�ne karakteristike filtra')
%set(gca,'YScale','log');
axis tight;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animacija premika impulz. odziva filtra v desno
LomnaFrek = 200;
Fsamp = 1024; % vzor�evalna frekvenca
F=[ones(1,LomnaFrek) zeros(1,Fsamp-2*LomnaFrek+1) ones(1,LomnaFrek-1)]; % nizko sito
f = ifftshift(ifft(F)); % zaradi periodi�nosti inverzne Fourierove transformiranke lahko
% �asovno os prestravimo za length(F)/2 vzorcev v levo.

close all; clc;
figure;
for dt = 0:5:200; % velikost premika v desno
    dRepL = 100; % dol�ina levega repa impuznega odziva;
    dRepD = 100; % dol�ina desnega repa impuznega odziva;
    f2 = f;
    f2(512-dRepL:1024)=f2(512-dt-dRepL:1024-dt); %pomik desno (naredimo filter vzro�en)
    f2(1:511+dt-dRepL)=0; % Postavljanje amplitud antivzro�nega dela na 0 (naredimo filter vzro�en in kon�en)
    f2(512+dt+dRepD:1024)=0; % pore�emo �e rep
    F2=fft(f2);  % FFT transformiranka novega impulznega odziva
    
    subplot(3,1,1); cla; hold on;
    plot([-length(f)/2:length(f)/2-1],f,'b');
    plot([-length(f2)/2:length(f2)/2-1],f2,'r');
    xlabel('t (vzorci)'); ylabel('amplituda');
    axis tight;
    title(['impulzni odziv frekven�nega filtra z Lomno frekvenco = ' num2str(LomnaFrek) ' Hz']);
 
    subplot(3,1,2); cla; hold on;
    plot(abs(F));
    plot(abs(F2),'r');
    xlabel('f (Hz)'); ylabel('amplituda');
    title('absolutna vrednost FFT')
    axis tight;
    
    subplot(3,1,3); cla; hold on;
    plot(unwrap(atan2(imag(F),real(F))),'LineWidth',2); % odvita faza;
    plot(unwrap(atan2(imag(F2),real(F2))),'r'); % odvita faza,'r';
    xlabel('f (Hz)'); ylabel('faza');
    title('faza FFT')
    axis tight;
    set(gca,'YLim',[0,2500]);
    
    pause(0.5);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animacija dol�ine impulz. odziva filtra - Gibbsov effekt
LomnaFrek = 200;
Fsamp = 1024; % vzor�evalna frekvenca
F=[ones(1,LomnaFrek) zeros(1,Fsamp-2*LomnaFrek+1) ones(1,LomnaFrek-1)]; % nizko sito
f = ifftshift(ifft(F)); % zaradi periodi�nosti inverzne Fourierove transformiranke lahko
% �asovno os prestravimo za length(F)/2 vzorcev v levo.

close all; clc;
figure;
dt = 100; % velikost premika v desno
for dRepL = 100:-5:10; % dol�ina levega repa impuznega odziva;
    
    dRepD = dRepL; % dol�ina desnega repa impuznega odziva;
    f2 = f;
    f2(512-dRepL:1024)=f2(512-dt-dRepL:1024-dt); %pomik desno (naredimo filter vzro�en)
    f2(1:511+dt-dRepL)=0; % Postavljanje amplitud antivzro�nega dela na 0 (naredimo filter vzro�en in kon�en)
    f2(512+dt+dRepD:1024)=0; % pore�emo �e rep
    F2=fft(f2);  % FFT transformiranka novega impulznega odziva
    
    subplot(3,1,1); cla; hold on;
    plot([-length(f)/2:length(f)/2-1],f,'b');
    plot([-length(f2)/2:length(f2)/2-1],f2,'r');
    xlabel('t (vzorci)'); ylabel('amplituda');
    axis tight;
    title(['impulzni odziv frekven�nega filtra z Lomno frekvenco = ' num2str(LomnaFrek) ' Hz, dol�ina filtra = ' num2str(dRepD + dRepL) ' vzorcev']);
 
    subplot(3,1,2); cla; hold on;
    plot(abs(F));
    plot(abs(F2),'r');
    xlabel('f (Hz)'); ylabel('amplituda');
    title('absolutna vrednost FFT')
    axis tight;
    
    subplot(3,1,3); cla; hold on;
    plot(unwrap(atan2(imag(F),real(F))),'LineWidth',2); % odvita faza;
    plot(unwrap(atan2(imag(F2),real(F2))),'r'); % odvita faza,'r';
    xlabel('f (Hz)'); ylabel('faza');
    title('faza FFT');
    axis tight;
    set(gca,'YLim',[0,2500]);
    
    pause(2);
end

'-------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% teko�e povpre�je

close all;  clc;
clear y;
fsamp = 1024; % frekvenca vzor�enja
x=randn(1,1*fsamp); % signal, ki ga �elimo filtrirati
m=2; % �t. koeficientov filtra oz. dol�ina filtra oz. red filtra (celo �tevilo) - MALO SE IGRAJTE 
for n=m:length(x)
   y(n)=sum(x(n-m+1:n));
end
X=fft(x);        % pretvorimo v frekven�ni prostor
Y=fft(y);        % pretvorimo v frekven�ni prostor

close all;  clc;
figure;

subplot(2,1,1);  hold on;
plot(x);         % izri�emo originalni signal
plot(y,'r','LineWidth',2);         % izri�emo rezultat filtriranja
xlabel('t (vzorci)'); ylabel('amplituda');
axis tight;
title(['Teko�e povpre�je, m = ' num2str(m)]);

subplot(2,1,2); hold on;
plot(abs(X));   % normalizirana FFT trasfromiranka signala X
plot(abs(Y),'r','LineWidth',2);     % frekven�na karakteristika filtra
xlabel('f (Hz)'); ylabel('amplituda');
title(['absolutna vrednost FFT']);
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animacija pri razli�nih redih fitra m
close all;  clc;
clear y;
fsamp = 1024; % frekvenca vzor�enja
x=randn(1,1*fsamp); % signal, ki ga �elimo filtrirati
X=fft(x);        % pretvorimo v frekven�ni prostor

figure;
for m=[1:4 5:5:100]; % red filtra
    for n=m:length(x)
        y(n)=mean(x(n-m+1:n));
    end
    Y=fft(y);        % pretvorimo v frekven�ni prostor
    
    subplot(2,1,1); cla; hold on;
    plot(x);         % izri�emo originalni signal
    plot(y,'r','LineWidth',2);         % izri�emo rezultat filtriranja
    xlabel('t (vzorci)'); ylabel('amplituda');
    axis tight;
    title(['Teko�e povpre�je, m = ' num2str(m)]);
    
    subplot(2,1,2); cla; hold on;
    plot(abs(X));   % normalizirana FFT trasfromiranka signala X
    plot(abs(Y),'r','LineWidth',2);     % frekven�na karakteristika filtra
    xlabel('f (Hz)'); ylabel('amplituda');
    title(['absolutna vrednost FFT']);
    axis tight;
    
    pause(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% frekven�na karakteristika teko�ega povpre�ja
close all;  clc;
figure;
for m=[1:4 5:5:20]; % red filtra
    subplot(2,1,1); cla; hold on;
    f=[ones(1,m) zeros(1,length(x)-m)]; % nizko sito
    F=fft(f);        % pretvorimo v frekven�ni prostor
    plot(abs(F),'m','LineWidth',2);     % frekven�na karakteristika filtra
    xlabel('f (Hz)'); ylabel('amplituda');
    title(['absolutna vrednost FFT, povpre�je preko, m = ' num2str(m)]);
    axis tight;
    
    subplot(2,1,2); cla; hold on;
    plot(unwrap(atan2(imag(F),real(F))),'LineWidth',2); % odvita faza;
    xlabel('f (Hz)'); ylabel('faza');
    title('faza FFT');
    axis tight;
    
    pause;
end

'------------------------------------------------------------------------------------------------------
% NA�RTOVANJE FILTROV V MATLABU
%    * butter   - Butterworth filter � no gain ripple in pass band and stop band, slow cutoff
%    * cheby1   - Chebyshev filter (Type I) � no gain ripple in stop band, moderate cutoff
%    * cheby2   - Chebyshev filter (Type II) � no gain ripple in pass band, moderate cutoff
%    * ellip    - Elliptic filter � gain ripple in pass and stop band, fast cutoff
%    * besself  - Bessel filter � no group delay ripple, no gain ripple in both bands, slow gain cutoff
'-------------------------------------------------------------------------------------------------------
 
clear;
clc

filter_order = 5; % red filtra (celo �tevilo) MALO SE IGRAJTE !!!!
Rp = 0.5; % 0.5 decibels of peak-to-peak ripple in the stopband/passband
Rp2 = 20; % 0.5 decibels of peak-to-peak ripple in the stopband/passband
Rs = 20; % minimum stopband attenuation of 20 decibels
Wn = 100/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz...
f = [0 0.2 0.2 1]; % frekven�ne to�ke
m = [1 1   0   0]; % ustrezne amplutide

subplot(2,2,1); hold on;
[b,a] = cheby1(filter_order,Rp,Wn,'low'); % izra�unamo koeficiente filtra
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f,m,'LineWidth',2);
plot(w/pi,abs(h),'r','LineWidth',2);
axis tight;
legend('idealni','na�rtovan');
xlabel('2*f/f_{vzor�evalna}','FontSize',12);
ylabel('amplituda','FontSize',12);
title(['Chebyshev filter (Tip 1): red =' num2str(filter_order)],'FontSize',12);

subplot(2,2,2); hold on;
[b,a] = cheby2(filter_order,Rp2,Wn,'low'); % izra�unamo koeficiente filtra
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f,m,'LineWidth',2);
plot(w/pi,abs(h),'r','LineWidth',2);
axis tight;
legend('idealni','na�rtovan');
xlabel('2*f/f_{vzor�evalna}','FontSize',12);
ylabel('amplituda','FontSize',12);
title(['Chebyshev filter (Tip 2): red =' num2str(filter_order)],'FontSize',12);

subplot(2,2,3); hold on;
[b,a] = butter(filter_order,Wn,'low'); % izra�unamo koeficiente filtra
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f,m,'LineWidth',2);
plot(w/pi,abs(h),'r','LineWidth',2);
axis tight;
legend('idealni','na�rtovan');
xlabel('2*f/f_{vzor�evalna}','FontSize',12);
ylabel('amplituda','FontSize',12);
title(['Butterworth filter: red =' num2str(filter_order)],'FontSize',12);

subplot(2,2,4); hold on;
[b,a] = ellip(filter_order,Rp,Rs,Wn,'low'); % izra�unamo koeficiente filtra
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f,m,'LineWidth',2);
plot(w/pi,abs(h),'r','LineWidth',2);
axis tight;
legend('idealni','na�rtovan');
xlabel('2*f/f_{vzor�evalna}','FontSize',12);
ylabel('amplituda','FontSize',12);
title(['Elliptic filter: red =' num2str(filter_order)],'FontSize',12);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POZOR! POZOR! POZOR! POZOR! POZOR!
%
% pri pasovno-prepustnih in pasovno-zapornih filtrih generirajo funkcije 
%    [b,a] = cheby1(filter_order,Rp,Wn)
%    [b,a] = cheby2(filter_order,Rp,Wn)
%    [b,a] = butter(filter_order,Wn)
% filtre reda 2*filter_order !!!
% glej help cheby1, help cheby2, help butter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHEBYSHEV FILTER TIPA 1 - no gain ripple in stop band, moderate cutoff
filter_order = 3; % red filtra
Rp = 0.5; % 0.5 decibels of peak-to-peak ripple in the passband
Wn = [100 200]/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz, zgornja lomna 200 Hz...
[b,a] = cheby1(filter_order,Rp,Wn); % izra�unamo koeficiente filtra
fvtool(b,a); % orodje za preu�evanje karakteristik filtra
 
figure;
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(w/pi,abs(h),'LineWidth',2);
axis tight;
legend('konstruiran filter');
xlabel('2*f/f_{vzor�evalna}','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['absolutna vrednost frekven�nega odziva, red filtra = ' num2str(filter_order)]);

originalni_signal = randn(1000,1);
filtriran_signal = filter(b,a,originalni_signal); % filtriramo signal "originalni_signal"
figure; subplot(2,1,1);
plot(originalni_signal); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('originalni signal');
subplot(2,1,2);
plot(filtriran_signal(:),'r'); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('filtriran signal');


% CHEBYSHEV FILTER TIPA 2  � no gain ripple in pass band, moderate cutoff
filter_order = 10; % red filtra
Rp2 = 100; % stopband ripple 20 dB down from the peak passband value
Wn = [100 200]/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz, zgornja lomna 200 Hz...
[b,a] = cheby2(filter_order,Rp2,Wn); % izra�unamo koeficiente filtra
fvtool(b,a); % orodje za preu�evanje karakteristik filtra

figure;
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(w/pi,abs(h),'LineWidth',2);
axis tight;
legend('konstruiran filter');
xlabel('2*f/f_{vzor�evalna}','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['absolutna vrednost frekven�nega odziva, red filtra = ' num2str(filter_order)]);

originalni_signal = randn(1000,1);
filtriran_signal = filter(b,a,originalni_signal); % filtriramo signal "originalni_signal"
figure; subplot(2,1,1);
plot(originalni_signal); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('originalni signal');
subplot(2,1,2);
plot(filtriran_signal,'r'); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('filtriran signal');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUTTERWORTH FILTER -  no gain ripple in pass band and stop band, slow cutoff
filter_order = 5; % red filtra
Wn = [100 200]/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz, zgornja lomna 200 Hz...
[b,a] = butter(filter_order,Wn); % izra�unamo koeficiente filtra
fvtool(b,a); % orodje za preu�evanje karakteristik filtra

figure;
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(w/pi,abs(h),'LineWidth',2);
axis tight;
legend('konstruiran filter');
xlabel('2*f/f_{vzor�evalna}','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['absolutna vrednost frekven�nega odziva, red filtra = ' num2str(filter_order)]);

originalni_signal = randn(10000,1);
filtriran_signal = filter(b,a,originalni_signal); % filtriramo signal "originalni_signal"
figure; subplot(2,1,1);
plot(originalni_signal); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('originalni signal');
subplot(2,1,2);
plot(filtriran_signal,'r'); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('filtriran signal');


'-------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIRLS FILTER - na�rtovanje FIR filtra s pomo�jo metode najmanj�ih kvadrati�nih pogre�kov (least-squares error minimization).
close all; clc;
filter_order = 25;
f = [0 0.3 0.4 0.6 0.7 1]; % frekven�ne to�ke 
m = [0 1   0   0   0.5 0.5]; % ustrezne amplutide 
b = firls(filter_order,f,m); % izra�unamo koeficiente b
a = 1; % firls ne vra�a koeficientov a, torej nastavimo a(1) na 1

fvtool(b,a);

figure;
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f,m,w/pi,abs(h),'LineWidth',2);
axis tight;
legend('Idealni','na�rtovan z metodo firls');
xlabel('2*f/f_{vzor�evalna}','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);

[y,t] = impz(b,a,101); % izra�unamo impulzni odziv filtra
figure; stem(t,y);  %prika�emo impulzni odziv filtra
axis tight; title('Impulzni odziv filtra'); 
xlabel('�as (vzorci)');
ylabel('amplituda');

fvtool(b,a); % orodje za preu�evanje karakteristik filtra

originalni_signal = randn(1000,1);
filtriran_signal = filter(b,a,originalni_signal); % filtriramo signal "originalni_signal"
figure; subplot(2,1,1);
plot(originalni_signal); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('originalni signal');
subplot(2,1,2);
plot(filtriran_signal,'r'); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('filtriran signal');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Yule-Walker FILTER - rekurzivna metoda najmanj�ih kvadrati�nih pogre�kov (recursive least-squares fit to a specified frequency response)
% close all; clc;
% filter_order = 10;
% m = [0   0   1   1   0   0   1   1   0 0];
% f = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1];
% [b,a] = yulewalk(filter_order,f,m);
% 
% figure;
% [h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
% plot(f,m,w/pi,abs(h),'LineWidth',2);
% axis tight;
% legend('Idealni','na�rtovan z metodo yulewalk');
% xlabel('2*f/f_{vzor�evalna}','FontSize',14); 
% ylabel('amplituda','FontSize',14);
% title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);
% 
% fvtool(b,a); % orodje za preu�evanje karakteristik filtra
% 
% originalni_signal = randn(1000,1);
% filtriran_signal = filter(b,a,originalni_signal); % filtriramo signal "originalni_signal"
% figure; subplot(2,1,1);
% plot(originalni_signal); axis tight; xlabel('vzorci'); ylabel('amplituda'); title('originalni signal');
% subplot(2,1,2);
% plot(filtriran_signal,'r'); axis tight; xlabel('vzorci');
% ylabel('amplituda'); title('filtriran signal');

'-------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% P R A K T I � N I      P R I M E R I
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'-------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% elektri�na kitara (https://ccrma.stanford.edu/~jos/waveguide/Sound_Examples.html)
[sig, Fs, nbits] = wavread('ElectricGuitar.wav');
%[sig, Fs, nbits] = wavread('gtr-dist-yes.wav');
sig = sig(:,1); % mono na�in (originalni signal je stereo)
disp(['Vzor�evalna frekvenca: ' num2str(Fs) ' Hz']);
disp(['Lo�ljivost: ' num2str(nbits) ' bitov']);
wavplay(sig,Fs,'sync')

close all;  clc;
tT = 1024; tN = 1024; dT = tT/2;
[TFD,F,T] = specgram(sig,tN,Fs,window(@hamming,tT),dT);
imagesc(T,F(1:128)/1000,abs(TFD(1:128,:))); axis xy;
xlabel('�as (s)'); ylabel('f (kHz)');
axis tight;
title(['originalni signal (�asovno-frekven�na ravnina)']);

% na�rtovanje filtra
filter_order = 10;
m = [1   1   0   0   0   0   0   0   0   0];  % nizko-prepustno sito
f = [0 0.01 0.02 0.3 0.4 0.5 0.6 0.7 0.8 1];
% m = [0   0   1   1   1   1   1   1   1   1];  % visoko-prepustno sito
% f = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1];

% Yule-Walker FILTER 
[b,a] = yulewalk(filter_order,f,m);
% FIRLS FILTER
% b = firls(filter_order,f,m); % izra�unamo koeficiente b
% a = 1; % firls ne vra�a koeficientov a, torej nastavimo a(1) na 1

% izris frekven�ne karakteristike filtra
figure;
[h,w] = freqz(b,a,128); % izris frekven�nih karakteristik filtra
plot(f*Fs/2000,m,w/pi*Fs/2000,abs(h),'LineWidth',2);
axis tight;
legend('Idealni','Implementirani');
xlabel('f (kHz)','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);

fvtool(b,a); % orodje za preu�evanje karakteristik filtra

% filtriramo signal
filt_sig = filter(b,a,sig);

% nari�emo TFD transformiranko
tT = 1024; tN = 1024; dT = tT/2;
figure; subplot(1,2,1) % originalni signal
[TFD,F,T] = specgram(sig,tN,Fs,window(@hamming,tT),dT);
imagesc(T,F(1:128)/1000,abs(TFD(1:128,:))); axis xy;
xlabel('�as (s)'); ylabel('f (kHz)');
axis tight;
title(['Originalni signal']);
subplot(1,2,2); % filtrirani signal
[TFD,F,T] = specgram(filt_sig,tN,Fs,window(@hamming,tT),dT);
imagesc(T,F(1:128)/1000,abs(TFD(1:128,:))); axis xy;
xlabel('�as (s)'); ylabel('f (kHz)');
axis tight;
title(['Filtrirani signal']);

% predvajajmo originalni in filtrirani signal
wavplay(4*filt_sig,Fs,'sync')
wavplay(1*sig,Fs,'sync')

% nari�emo originalni in filtrirani signal
figure; hold on; 
plot(sig); 
plot(filt_sig,'r');
title(['Opazujte zamik med originalnim in filtriranim signalom pri filtru reda ' num2str(filter_order)]);
axis tight;

% namesto funkcije filter uprabimo funkcijo filtfilt
filtfilt_sig = filtfilt(b,a,sig);
figure; hold on; 
plot(sig); 
plot(filtfilt_sig,'g','LineWidth',2);
title(['filtfilt: je �e vedno zamik med originalnim in filtriranim signalom? Red filtra = ' num2str(filter_order)]);
axis tight;

wavplay(1*filtfilt_sig,Fs,'sync')
wavplay(1*filt_sig,Fs,'sync')

'-----------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linearnost faze

clear; close all; clc;

% FIRLS FILTER
filter_order = 30;
m = [0   0   1   1   0   0];
f = [0 0.2 0.2 0.4   0.4 1];
[b1,a1] = firls(filter_order,f,m);

fvtool(b1,a1)
% izris frekven�ne karakteristike filtra
figure;
[h,w] = freqz(b1,a1,128); % izris frekven�nih karakteristik filtra
plot(f,m,w/pi,abs(h),'LineWidth',2);
axis tight;
legend('Idealni','Implementirani');
xlabel('f (kHz)','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);
% end of FIRLS

% CHEBYSHEV FILTER TIPA 1  � no gain ripple in stop band, moderate cutoff
filter_order = 10; % red filtra
Rp1 = 0.5; % % 0.5 decibels of peak-to-peak ripple in the passband
Wn = [100 200]/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz, zgornja lomna 200 Hz...
[b2,a2] = cheby1(filter_order,Rp1,Wn); % izra�unamo koeficiente filtra

fvtool(b2,a2); % orodje za preu�evanje karakteristik filtra
% izris frekven�ne karakteristike filtra
figure;
[h,w] = freqz(b2,a2,128); % izris frekven�nih karakteristik filtra
plot(f,m,w/pi,abs(h),'LineWidth',2);
axis tight;
legend('Idealni','Implementirani');
xlabel('f (kHz)','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);
% end of CHEBYSHEV FILTER TIPA 1

% BUTTERWORTH FILTER -  no gain ripple in pass band and stop band, slow cutoff
filter_order = 10; % red filtra
Wn = [100 200]/500; % definicija lomnih frekvenc (vzor�evalna frevenca 1000 Hz, spodnja lomna 100 Hz, zgornja lomna 200 Hz...
[b3,a3] = butter(filter_order,Wn); % izra�unamo koeficiente filtra

fvtool(b3,a3); % orodje za preu�evanje karakteristik filtra
% izris frekven�ne karakteristike filtra
figure;
[h,w] = freqz(b3,a3,128); % izris frekven�nih karakteristik filtra
plot(f,m,w/pi,abs(h),'LineWidth',2);
axis tight;
legend('Idealni','Implementirani');
xlabel('f (kHz)','FontSize',14); 
ylabel('amplituda','FontSize',14);
title(['Primerjava absolutnih vrednosti frekven�nega odziva, red filtra = ' num2str(filter_order)]);
% end of BUTTERWORTH FILTER

% mexican hat
t=-10:0.5:10;
mh=exp((-t.^2)/2).*(1-t.^2); % ustvarimo signal - mexican hat
sig=zeros(1,1024);
sig(200:200+length(mh)-1)=mh;  % na za�etek in konec signala dodam ni�le

figure;
subplot(2,2,1); plot(sig); axis tight; title('originalni signal');
subplot(2,2,2); plot(filter(b1,a1,sig)); axis tight; title('filtriran signal (firls)');
subplot(2,2,3); plot(filter(b2,a2,sig)); axis tight; title('filtriran signal (cheby1)');
subplot(2,2,4); plot(filter(b3,a3,sig)); axis tight; title('filtriran signal (butter)');
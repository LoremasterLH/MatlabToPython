'--------------------------------------------------------------------------------------
% �asovno-frekven�na lo�ljivost in �asovno-frekven�ne porazdelitve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
path('.\tftb\tftb-0.2\tftb-0.2\mfiles',path); % pot do �asovno-frekven�nega toolbox-a

Fs = 1024;
sig=hilbert(fmlin(1*Fs)); % analiti�en signal s linearno nara��ajo�o frekvenco. 

figure; plot(real(sig)); % nari�emo realni del v �asovni domeni (signal je kompleksen, a le zaradi umetno zagotovljene analiti�nosti)
xlabel('t'); ylabel('amplituda');
axis tight;

figure; plot(abs(fft(sig))); % analiti�ni signali nimajo simetrije preko Nyquistove frekvence
xlabel('f'); ylabel('amplituda');
axis tight;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �asovno-frekven�na lo�ljivost prvi�: vi�ja �asovna lo�ljivost, ni�ja frekven�na lo�ljivost
close all
T = 32;
N = 256;
dT = T-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
axis tight;
title(['T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

% �asovno-frekven�na lo�ljivost drugi�: ni�ja �asovna lo�ljivost, vi�ja frekven�na lo�ljivost
T = 256;
N = 256;
dT = T-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
axis tight;
title(['T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

% Wigner-Villova �asovno-frekven�na porazdelitev - skoraj idealna �asovna in frekven�na lo�ljivost
[TFD,t,f] = tfrwv(sig);
figure; 
imagesc(t,f,TFD); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight; axis xy;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);


% Trenutna autokorelacija  Rss(t,tau) = sig(t+tau) .* conj(sig(t-tau)); -  kdavdratna funkcija signala---------------------
%   je osnova za Wigner-Villovo �asovnofrekven�no porazdelitev in omogo�a skoraj idealno �asovno in frekven�no lo�ljivost
t=1:length(sig); % dol�ina signala v vzorcih
tcol = length(sig); % za vsako �asovni trenutek (vsak vzorec)
Rss= zeros (tcol,tcol); % izra�unamo trenutno pre�no korelacijo.
for icol=1:tcol,
    ti = t(icol); % trenutna pozicija / trenutni �as 
    taumax=min([ti-1,tcol-ti,round(tcol/2)-1]); % maksimalen zamik 
    
    tau=-taumax:taumax; % vektor vsem mo�nih zamikov
    indices= rem(tcol+tau,tcol)+1; % mesto v Rss, kjer se bodo shranili rezultati
    Rss(indices,icol) = sig(ti+tau) .* conj(sig(ti-tau)); % izra�un trenutne pre�ne korelacije
end;

figure; % izris rezultatov
for icol=1:10:tcol/2 % risali bomo vsak 10 izra�un (�asovni korak 10 vzorcev)
    cla;
    subplot(2,1,1); plot(real(Rss(:,icol))); title(['t = ' num2str(icol) ', real Rss']); axis tight;
    subplot(2,1,2); plot(imag(Rss(:,icol))); title(['t = ' num2str(icol) ', imag Rss']); axis tight;
    pause;
end;

% Wigner-Villova �asovno-frekven�na porazdelitev - skoraj idealna �asovna in frekven�na lo�ljivost----------------------------
% Wigner-Villova �asovno-frekven�na porazdelitev je Fourierova transformacija trenutne pre�ne korelacije Rss.
[TFD,t,f] = tfrwv(sig); % funkcija tfrwv je implementirana v toolbox-u tftb
figure; 
contour(t,f,TFD); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);

'------------------------------------------------------------------------------------------------------------------------------
% �asovno-frekven�na lo�ljivost in �asovno-frekven�ne porazdelitve - ve� signalov skupaj
% Primer 1: odkomentirajte za zgled z dvema asimetri�no razporejenima atomoma v �asovno-frekven�ni ravnini. 
Fs = 1024;
sig=atoms(1*Fs,[1*Fs/4,.15,20,1; ...   % Deli signala, ki imajo lo�eno �asovno-frekven�no podporo se imenujejo atomi
                3*Fs/4,.35,20,1; ...   % Ukaz atoms, generira poljubno �tevilo atomov, v na�em primeru 2.
                                ]);    % Prvi argument je center v �asu, drugi center v frekvenci, tretji dol�ina, �eterti frekven�na �irina    

            
figure; plot(real(sig)); % nari�emo realni del v �asovni domeni (signal je kompleksen, a le zaradi umetno zagotovljene analiti�nosti)
xlabel('t'); ylabel('amplituda');
axis tight;

figure; plot(abs(fft(sig))); % analiti�ni signali nimajo simetrije preko Nyquistove frekvence
xlabel('f'); ylabel('amplituda');
axis tight;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �asovno-frekven�na lo�ljivost prvi�: vi�ja �asovna lo�ljivost, ni�ja frekven�na lo�ljivost---------------------------------
close all
T = 32;
N = 256;
dT = T-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
axis tight;
title(['Spektrogram: T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

% �asovno-frekven�na lo�ljivost drugi�: ni�ja �asovna lo�ljivost, vi�ja frekven�na lo�ljivost--------------------------------
T = 256;
N = 256;
dT = T-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
axis tight;
title(['Spektrogram: T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

% Wigner-Villova �asovno-frekven�na porazdelitev - skoraj idealna �asovna in frekven�na lo�ljivost----------------------------
[TFD,t,f] = tfrwv(sig);
figure; 
contour(t,f,abs(TFD),100); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);

% Wigner-Villova �asovno-frekven�na porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!! ---------------------------
[TFD,t,f] = tfrwv(real(sig));
figure; 
contour(t,f,abs(TFD),100); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);


% pseudo-Wigner-Villova �asovno-frekven�na porazdelitev -----------------------------------------------------------------------
%    - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
[TFD,t,f] = tfrpwv(sig,1:Fs,Fs,hamming(31)); 
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Psevdo Wigner-Villova �asovno-frekven�na porazdelitev']);

% pseudo-Wigner-Villova �asovno-frekven�na porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4  !!! -----------------------------------------
%    - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
[TFD,t,f] = tfrpwv(real(sig),1:Fs,Fs,hamming(31)); 
figure; 
contour(t,f,abs(TFD)); axis tight;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Psevdo Wigner-Villova �asovno-frekven�na porazdelitev']);



'------------------------------------------------------------------------------------------------------------------------------
% �asovno-frekven�na lo�ljivost in �asovno-frekven�ne porazdelitve - ve� signalov skupaj

% % Primer 2: zgled z �tirimi simetri�no razporejenimi atomi v �asovno-frekven�ni ravnini.             
Fs = 1024;
sig=atoms(1*Fs,[1*Fs/4,.15,20,1; ...   % Deli signala, ki imajo lo�eno �asovno-frekven�no podporo se imenujejo atomi
                3*Fs/4,.15,20,1; ...   % Ukaz atoms, generira poljubno �tevilo atomov, v na�em primeru 4.
                1*Fs/4,.35,20,1; ...   % Prvi argument je center v �asu, drugi center v frekvenci, tretji dol�ina, �eterti frekven�na �irina
                3*Fs/4,.35,20,1]);     


figure; plot(real(sig)); % nari�emo realni del v �asovni domeni (signal je kompleksen, a le zaradi umetno zagotovljene analiti�nosti)
xlabel('t'); ylabel('amplituda');
axis tight;

figure; plot(abs(fft(sig))); % analiti�ni signali nimajo simetrije preko Nyquistove frekvence
xlabel('f'); ylabel('amplituda');
axis tight;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �asovno-frekven�na lo�ljivost prvi�: vi�ja �asovna lo�ljivost, ni�ja frekven�na lo�ljivost---------------------------------
close all
T = 32;
N = 256;
dT = T-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
figure; 
imagesc(t,f(1:end/2),abs(TFD(1:end/2,:))); axis xy;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
title(['Spektrogram: T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

    % �asovno-frekven�na lo�ljivost drugi�: ni�ja �asovna lo�ljivost, vi�ja frekven�na lo�ljivost--------------------------------
    T = 256;
    N = 256;
    dT = T-1;
    [TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,T),dT);
    figure; 
    imagesc(t,f(1:end/2),abs(TFD(1:end/2,:))); axis xy;
    xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
    axis tight;
    title(['Spektrogram: T=' num2str(T) ',N=' num2str(N) ',dT=' num2str(dT)]);

% Wigner-Villova �asovno-frekven�na porazdelitev - skoraj idealna �asovna in frekven�na lo�ljivost----------------------------
[TFD,t,f] = tfrwv(sig);
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);

% pseudo-Wigner-Villova �asovno-frekven�na porazdelitev -----------------------------------------------------------------------
%    - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
[TFD,t,f] = tfrpwv(sig,1:Fs,Fs,hamming(31)); 
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Psevdo Wigner-Villova �asovno-frekven�na porazdelitev']);
   
% pseudo-Wigner-Villova �asovno-frekven�na porazdelitev - SAMO REALNI DEL SIGNALA - Zrcaljenje preko frekce Fsamp/4 !!!! ----------------------------------------
%    - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
[TFD,t,f] = tfrpwv(real(sig),1:Fs,Fs,hamming(31)); 
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Psevdo Wigner-Villova �asovno-frekven�na porazdelitev']);


% Choi-Williams �asovno-frekven�na porazdelitev ------------------------------------------------------------------------------  
%      - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
%      - okno v frekven�ni domeni, ki prepre�uje interference med frekven�no odmaknjenimi atomi    
[TFD,t,f] = tfrcw(sig,1:Fs,Fs,hamming(31),hamming(31));
figure; 
imagesc(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Choi-Williams �asovno-frekven�na porazdelitev']);

'-----------------------------------------------------------------------------------------------
% �asovno-frekven�ne porazdelitve: primeri za igranje in u�enje
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
T = 1; % trajanje signala v s
Fs = 512; % vzor�evalna frekvenca

% Primer 1: analiti�en signal z linearno nara��ajo�o frekvenco. 
[sig,trenutnaFrekvenca]=fmlin(T*Fs);
sig=hilbert(sig); % naredimo signal analiti�en (�e slu�ajno �e ni)

% Primer 2: analiti�en signal s sinusoidno spremeinjajo�o se frekvenco. 
[sig,trenutnaFrekvenca]=fmsin(T*Fs,0.05,0.45,100,20,0.3,-1.0);
sig=hilbert(sig); % naredimo signal analiti�en (�e slu�ajno �e ni)

% Primer 3: signal s eksponentno spremeinjajo�o se frekvenco. 
[sig,trenutnaFrekvenca]=fmpower(T*Fs,0.5,[1 0.5],[100 0.1]);
sig=hilbert(sig); % naredimo signal analiti�en (�e slu�ajno �e ni)

% Od tu dalje je je koda za katerikoli zgornji primer.
figure; subplot(2,1,1); % nari�emo realni del v �asovni domeni (signal je kompleksen, a le zaradi umetno zagotovljene analiti�nosti) 
plot([0:length(sig)-1]/Fs,real(sig)); axis tight;
xlabel('t (s)'); ylabel('amplituda'); title('signal')
 
subplot(2,1, 2); % nari�emo njegovo (simulirano) trenutno frekvenco
plot([0:length(sig)-1]/Fs,trenutnaFrekvenca*Fs); xlabel('t (s)'); ylabel('f (Hz)'); title('trenutna frekvenca'); axis tight;

% �asovno-frekven�na lo�ljivost prvi�: vi�ja �asovna lo�ljivost, ni�ja frekven�na lo�ljivost---------------------------------
close all
winT = 32;
N = 256;
dT = winT-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,winT),dT);
figure; 
contour(t,f(1:end/2),abs(TFD(1:end/2,:))); axis xy;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
title(['Spektrogram: winT=' num2str(winT) ',N=' num2str(N) ',dT=' num2str(dT)]);

% �asovno-frekven�na lo�ljivost drugi�: ni�ja �asovna lo�ljivost, vi�ja frekven�na lo�ljivost--------------------------------
winT = 256;
N = 256;
dT = winT-1;
[TFD,f,t] = specgram(sig,N,Fs,window(@rectwin,winT),dT);
figure; 
imagesc(t,f(1:end/2),abs(TFD(1:end/2,:))); axis xy;
xlabel('t (s)','FontSize',12); ylabel('f (Hz)','FontSize',12)
axis tight;
title(['Spektrogram: winT=' num2str(winT) ',N=' num2str(N) ',dT=' num2str(dT)]);

% Wigner-Villova �asovno-frekven�na porazdelitev - skoraj idealna �asovna in frekven�na lo�ljivost----------------------------
[TFD,t,f] = tfrwv(sig,1:T*Fs,Fs);
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Wigner-Villova �asovno-frekven�na porazdelitev']);

% pseudo-Wigner-Villova �asovno-frekven�na porazdelitev -----------------------------------------------------------------------
%    - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
[TFD,t,f] = tfrpwv(sig,1:T*Fs,Fs,hamming(31)); 
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Psevdo Wigner-Villova �asovno-frekven�na porazdelitev']);
             
% Choi-Williams �asovno-frekven�na porazdelitev ------------------------------------------------------------------------------  
%      - okno v �asovni domeni, ki prepre�uje interference med �asovno odmaknjenimi atomi
%      - okno v frekven�ni domeni, ki prepre�uje interference med frekven�no odmaknjenimi atomi    
[TFD,t,f] = tfrcw(sig,1:T*Fs,Fs,hamming(25),hamming(25));
figure; 
contour(t,f,abs(TFD)); axis xy;
xlabel('t','FontSize',12); ylabel('f','FontSize',12)
axis tight;
title(['Choi-Williams �asovno-frekven�na porazdelitev']);

'------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Primeri iz realnega �ivljenja
clc;
clear;
close all
Fs = 8000; % vzor�evalna frekvenca
bits = 8; % bitna lo�ljivost
T = 2; % �as snemanja v sekundah

my_recorder = audiorecorder(Fs,bits,1);

disp(['Snemam posnetek v dol�ini ' num2str(T) 's...']);
recordblocking(my_recorder,T);
sig = getaudiodata(my_recorder);
figure; plot([1:T*Fs]/Fs,sig); title('Posneti signal'); xlabel('t (s)'); ylabel('amplituda');
wavplay(sig,Fs,'sync');

sig=hilbert(sig); % naredimo signal analiti�en (zvo�ni posnetek zagotovo ni)

% od tu dalje je pa dodajte kodo sami. Govorimo seveda o �asovno-frekven�ni analizi.
% POZOR! �asovno-frekven�ne porazdelitve so lahko ra�unsko precej precej po�re�ne.
% Priporo�am majhne korake pri pove�avi dol�ine signala in smelo nastavljanje �tevila frekven�nih to�k.
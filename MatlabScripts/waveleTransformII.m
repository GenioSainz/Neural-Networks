% mikexcohen@gmail.com
clc
load sampleEEGdata.mat

DC_value  = 0;
F         = [25];
A         = [1];

EEG.pnts  = 1000;
EEG.srate = 100;
EEG.times = (0:EEG.pnts-1)*(1/EEG.srate);

% multiple sines
% EEG.data  = getComposeSignal(F,A,DC_value,EEG.times)+chirp(EEG.times,25,10,30,'quadratic');

% chirps
t1 = EEG.times(end);
EEG.data = getComposeSignal(F,A,DC_value,EEG.times)+...
           chirp1(EEG.times,35,t1,45)+...
           chirp2(EEG.times,5,t1,15);

% wavelet parameters
num_frex = 1000;
min_freq = 1e-6;
max_freq = 50;

% other wavelet parameters
frex      = linspace(min_freq,max_freq,num_frex);
time      = -2:1/EEG.srate:2;  
half_wave = (length(time)-1)/2;

% FFT parameters
nKern = length(time);
nData = EEG.pnts;
nConv = nKern+nData-1;
dataX = fft(EEG.data,nConv);

% set a few different wavelet widths (number of wavelet cycles)
range_cycles = [16 32];
nCycles      = linspace(range_cycles(1),range_cycles(2),num_frex);


% initialize output time-frequency data
tf = zeros(length(frex),EEG.pnts);

for fi=1:num_frex
    
    % create wavelet and get its FFT
    n   = 16;
    n   = nCycles(fi);
    s   = n/(2*pi*frex(fi));
    cmw = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
    
    cmwX = fft(cmw,nConv);
    cmwX = cmwX./max(cmwX);

    % run convolution
    as = ifft(cmwX.*dataX,nConv);
    as = as(half_wave+1:end-half_wave);
    %as = reshape(as,EEG.pnts,EEG.trials);
    
    % put power data into big matrix
    tf(fi,:) = abs(as).^2;
end

figure(1), clf

plot(EEG.times,EEG.data)
xlim([0 1])
figure(2), clf

subplot(1,2,1)
    hold on;
    pbaspect([1 1 1])
    xlabel('Time (s) '), ylabel('Frequency (Hz)')
    set(gca,'YDir','normal')

    max_x = max(tf(:));
    mea_x = mean(tf(:));
    min_x = min(tf(:));
    title(sprintf('Max:%0.8f Mean:%0.8f',max_x,mea_x))
    imagesc(EEG.times,frex,tf)
    %plot(EEG.times,f,'r')
    axis([min(EEG.times) max(EEG.times) min_freq max_freq]);

subplot(1,2,2)
    hold on;box on;grid on
    pbaspect([1 1 1])
    xlabel('Frequency (Hz)')
    ylabel('Magnitude')
    Y = abs(fft(EEG.data))/EEG.pnts;
    hz = (0:EEG.pnts/2 + 1)*EEG.srate/EEG.pnts;
    Y  = 2*Y(1:length(hz));
    plot(hz,Y)
    ylim([0,1])
    xlim([0,50])
    xticks(0:5:50)

function y = getComposeSignal(F,A,DC_value,t)

    N = length(t);
    y = zeros(1,N) + DC_value;

    for k=1:length(A)

        y = y + A(k)*sin(2*pi*F(k)*t);
    end
end


function  y = chirp1(t,f0,t1,f1)

          c =(f1-f0)/t1;
          f = f0+c*t./2;
          y = cos( 2*pi*t.*f );
end

function  y = chirp2(t,f0,t1,f1)
          
          c = (f1-f0)/t1^2;
          f = f0+c*t.^2/3;
          y = cos(2*pi*t.*f); 
end


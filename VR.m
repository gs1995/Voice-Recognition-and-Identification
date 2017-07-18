function []=VR()
%% 1 Voice Recognition by letting user enter into database and then compare
%Voice recognition is the process of automatically recognizing who is
%speaking on the basis of individual information included in speech waves.
%
%This technique makes it possible to use the speaker's voice to verify
%their identity and control access to services such as voice dialing,
%telephone shopping, database access services, information
%services, voice mail, and remote access to computers.

ronaldo=10;


disp('> 10: Test with other speech files')
msgbox('P.S. This prototype is for secondary security usage.','NOTE','help');
pause(2);
msgbox('Kindly Note this works for the stored databases only. This means that you can add sounds to the database by users and Recognition will be done for the users entered. ','NOTE','help')
pause(2);

chos=0;
possibility=5;
while chos~=possibility,
    chos=menu('Speaker Recognition System','Add a new sound from microphone',...
        'Speaker recognition from microphone',...
        'Database Info','Delete database','Exit');
    %----------------------------------------------------------------------
    
    %% 1.1 Add a new sound from microphone
    
    if chos==1
        
        if (exist('sound_database.dat','file')==2)
            load('sound_database.dat','-mat');
            classe = input('Insert a class number (sound ID) that will be used for recognition:');
            if isempty(classe)
                classe = sound_number+1;
                disp( num2str(classe) );
            end
            message=('The following parameters will be used during recording:');
            disp(message);
            message=strcat('Sampling frequency',num2str(samplingfrequency));
            disp(message);
            message=strcat('Bits per sample',num2str(samplingbits));
            disp(message);
            durata = input('Insert the duration of the recording (in seconds):');
            if isempty(durata)
                durata = 3;
                disp( num2str(durata) );
            end
            micrecorder = audiorecorder(samplingfrequency,samplingbits,1);
            disp('Now, speak into microphone...');
            record(micrecorder,durata);
            
            while (isrecording(micrecorder)==1)
                disp('Recording...');
                pause(0.5);
            end
            disp('Recording stopped.');
            y1 = getaudiodata(micrecorder);
            y = getaudiodata(micrecorder, 'uint8');
            
            if size(y,2)==2
                y=y(:,1);
            end
            y = double(y);
            sound_number = sound_number+1;
            data{sound_number,1} = y;
            data{sound_number,2} = classe;
            data{sound_number,3} = 'Microphone';
            data{sound_number,4} = 'Microphone';
            st=strcat('u',num2str(sound_number));
            wavwrite(y1,samplingfrequency,samplingbits,st)
            save('sound_database.dat','data','sound_number','-append');
            msgbox('Sound added to database','Database result','help');
            disp('Sound added to database');
            
        else
            classe = input('Insert a class number (sound ID) that will be used for recognition:');
            if isempty(classe)
                classe = 1;
                disp( num2str(classe) );
            end
            durata = input('Insert the duration of the recording (in seconds):');
            if isempty(durata)
                durata = 3;
                disp( num2str(durata) );
            end
            samplingfrequency = input('Insert the sampling frequency (22050 recommended):');
            if isempty(samplingfrequency )
                samplingfrequency = 22050;
                disp( num2str(samplingfrequency) );
            end
            samplingbits = input('Insert the number of bits per sample (8 recommended):');
            if isempty(samplingbits )
                samplingbits = 8;
                disp( num2str(samplingbits) );
            end
            micrecorder = audiorecorder(samplingfrequency,samplingbits,1);
            disp('Now, speak into microphone...');
            record(micrecorder,durata);
            
            while (isrecording(micrecorder)==1)
                disp('Recording...');
                pause(0.5);
            end
            disp('Recording stopped.');
            y1 = getaudiodata(micrecorder);
            y = getaudiodata(micrecorder, 'uint8');
            
            if size(y,2)==2
                y=y(:,1);
            end
            y = double(y);
            sound_number = 1;
            data{sound_number,1} = y;
            data{sound_number,2} = classe;
            data{sound_number,3} = 'Microphone';
            data{sound_number,4} = 'Microphone';
            st=strcat('u',num2str(sound_number));
            wavwrite(y1,samplingfrequency,samplingbits,st)
            save('sound_database.dat','data','sound_number','samplingfrequency','samplingbits');
            msgbox('Sound added to database','Database result','help');
            disp('Sound added to database');
        end
    end
    
    %----------------------------------------------------------------------
    
    %% 1.2 Voice Recognition from microphone
    
    if chos==2
        
        if (exist('sound_database.dat','file')==2)
            load('sound_database.dat','-mat');
            Fs = samplingfrequency;
            durata = input('Insert the duration of the recording (in seconds):');
            if isempty(durata)
                durata = 3;
                disp( num2str(durata) );
            end
            micrecorder = audiorecorder(samplingfrequency,samplingbits,1);
            disp('Now, speak into microphone...');
            record(micrecorder,durata);
            
            while (isrecording(micrecorder)==1)
                disp('Recording...');
                pause(0.5);
            end
            disp('Recording stopped.');
            y = getaudiodata(micrecorder);
            st='v';
            wavwrite(y,samplingfrequency,samplingbits,st);
            y = getaudiodata(micrecorder, 'uint8');
            % if the input sound is not mono
            
            if size(y,2)==2
                y=y(:,1);
            end
            y = double(y);
            %----- code for speaker recognition -------
            disp('MFCC cofficients computation and VQ codebook training in progress...');
            disp(' ');
            % Number of centroids required
            k =16;
            
            for ii=1:sound_number
                % Compute MFCC cofficients for each sound present in database
                v = mfcc(data{ii,1}, Fs);
                % Train VQ codebook
                code{ii} = vqlbg(v, k);
                disp('...');
            end
            disp('Completed.');
            % Compute MFCC coefficients for input sound
            v = mfcc(y,Fs);
            % Current distance and sound ID initialization
            distmin = Inf;
            k1 = 0;
            
            for ii=1:sound_number
                d = disteu(v, code{ii});
                dist = sum(min(d,[],2)) / size(d,1);
                message=strcat('For User #',num2str(ii),' Dist : ',num2str(dist));
                disp(message);
                
                if dist < distmin
                    distmin = dist;
                    k1 = ii;
                end
            end
            
            if distmin < ronaldo
                min_index = k1;
                speech_id = data{min_index,2};
                %-----------------------------------------
                disp('Matching sound:');
                message=strcat('File:',data{min_index,3});
                disp(message);
                message=strcat('Location:',data{min_index,4});
                disp(message);
                message = strcat('Recognized speaker ID: ',num2str(speech_id));
                disp(message);
                msgbox(message,'Matching result','help');
                
                ch3=0;
                while ch3~=3
                    ch3=menu('Matched result verification:','Recognized Sound','Recorded sound','Exit');
                    
                    if ch3==1
                        st=strcat('u',num2str(speech_id));
                        [s fs nb]=wavread(st);
                        p=audioplayer(s,fs,nb);
                        play(p);
                    end
                    
                    if ch3==2
                        [s fs nb]=wavread('v');
                        p=audioplayer(s,fs,nb);
                        play(p);
                    end
                end
                
            else
                warndlg('Wrong User . No matching Result.',' Warning ')
            end
        else
            warndlg('Database is empty. No matching is possible.',' Warning ')
        end
    end
    %----------------------------------------------------------------------
    
    %% 1.3 Database Info
    
    if chos==3
        
        if (exist('sound_database.dat','file')==2)
            load('sound_database.dat','-mat');
            message=strcat('Database has #',num2str(sound_number),'words:');
            disp(message);
            disp(' ');
            
            for ii=1:sound_number
                message=strcat('Location:',data{ii,3});
                disp(message);
                message=strcat('File:',data{ii,4});
                disp(message);
                message=strcat('Sound ID:',num2str(data{ii,2}));
                disp(message);
                disp('-');
            end
            
            ch32=0;
            while ch32 ~=2
                ch32=menu('Database Information','Database','Exit');
                
                if ch32==1
                    st=strcat('Sound Database has : #',num2str(sound_number),'words. Enter a database number : #');
                    prompt = {st};
                    dlg_title = 'Database Information';
                    num_lines = 1;
                    def = {'1'};
                    options.Resize='on';
                    options.WindowStyle='normal';
                    options.Interpreter='tex';
                    an = inputdlg(prompt,dlg_title,num_lines,def);
                    an=cell2mat(an);
                    a=str2double(an);
                    
                    if (isempty(an))
                        
                    else
                        
                        if (a <= sound_number)
                            st=strcat('u',num2str(an));
                            [s fs nb]=wavread(st);
                            p=audioplayer(s,fs,nb);
                            play(p);
                        else
                            warndlg('Invalid Word ','Warning');
                        end
                    end
                end
            end
            
        else
            warndlg('Database is empty.',' Warning ')
        end
    end
    %----------------------------------------------------------------------
    
    %% 1.4 Delete database
    
    if chos==4
        %clc;
        close all;
        
        if (exist('sound_database.dat','file')==2)
            button = questdlg('Do you really want to remove the Database?');
            
            if strcmp(button,'Yes')
                load('sound_database.dat','-mat');
                
                for ii=1:sound_number
                    st=strcat('u',num2str(ii),'.wav');
                    delete(st);
                end
                
                if (exist('v.wav','file')==2)
                    delete('v.wav');
                end
                
                delete('sound_database.dat');
                msgbox('Database was succesfully removed from the current directory.','Database removed','help');
            end
            
        else
            warndlg('Database is empty.',' Warning ')
        end
    end
end
end
        %--------------------------------------------------------------------------
        
        %% MFCC Function
        % MFCC
        %
        % Inputs: s contains the signal to analyze
        % fs is the sampling rate of the signal
        %
        % Output: r contains the transformed signal
        
        function r = mfcc(s, fs)
        m = 100;
        n = 256;
        frame=blockFrames(s, fs, m, n);
        m = melfb(20, n, fs);
        n2 = 1 + floor(n / 2);
        z = m * abs(frame(1:n2, :)).^2;
        r = dct(log(z));
        end
        %--------------------------------------------------------------------------

    %% blockFrames Function
    % blockFrames: Puts the signal into frames
    %
    % Inputs: s contains the signal to analize
    % fs is the sampling rate of the signal
    % m is the distance between the beginnings of two frames
    % n is the number of samples per frame
    %
    % Output: M3 is a matrix containing all the frames

    function M3 = blockFrames(s, fs, m, n)
    l = length(s);
    nbFrame = floor((l - n) / m) + 1;
    for i = 1:n
        for j = 1:nbFrame
            M(i, j) = s(((j - 1) * m) + i); %#ok<AGROW>
        end
    end
    h = hamming(n);
    M2 = diag(h) * M;
    for i = 1:nbFrame
        M3(:, i) = fft(M2(:, i)); %#ok<AGROW>
    end
    end
    %--------------------------------------------------------------------------
function m = melfb(p, n, fs)
% MELFB Determine matrix for a mel-spaced filterbank
%
% Inputs: p number of filters in filterbank
% n length of fft
% fs sample rate in Hz
%
% Outputs: x a (sparse) matrix containing the filterbank amplitudes
% size(x) = [p, 1+floor(n/2)]
%
% Usage: For example, to compute the mel-scale spectrum of a
% column-vector signal s, with length n and sample rate fs:
%
% f = fft(s);
% m = melfb(p, n, fs);
% n2 = 1 + floor(n/2);
% z = m * abs(f(1:n2)).^2;
%
% z would contain p samples of the desired mel-scale spectrum
%
% To plot filterbanks e.g.:
%
% plot(linspace(0, (12500/2), 129), melfb(20, 256, 12500)'),
% title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');
%%%%%%%%%%%%%%%%%%
%
f0 = 700 / fs;
fn2 = floor(n/2);
lr = log(1 + 0.5/f0) / (p+1);
% convert to fft bin numbers with 0 for DC term
bl = n * (f0 * (exp([0 1 p p+1] * lr) - 1));
b1 = floor(bl(1)) + 1;
b2 = ceil(bl(2));
b3 = floor(bl(3));
b4 = min(fn2, ceil(bl(4))) - 1;
pf = log(1 + (b1:b4)/n/f0) / lr;
fp = floor(pf);
pm = pf - fp;
r = [fp(b2:b4) 1+fp(1:b3)];
c = [b2:b4 1:b3] + 1;
v = 2 * [1-pm(b2:b4) pm(1:b3)];
m = sparse(r, c, v, p, 1+fn2);
end
%-------------------------------------------------------------------------- 
%--------------------------------------------------------------------------

%% VQLBG Vector quantization using the Linde-Buzo-Gray algorithm
% VQLBG Vector quantization using the Linde-Buzo-Gray algorithm
%
% Inputs: d contains training data vectors (one per column)
% k is number of centroids required
%
% Output: r contains the result VQ codebook (k columns, one for each  centroids)

function r = vqlbg(d,k)
e = .01;
r = mean(d, 2);
dpr = 10000;
for i = 1:log2(k)
    r = [r*(1+e), r*(1-e)];
    while (1 == 1)
        z = disteu(d, r);
        [m,ind] = min(z, [], 2);
        t = 0;
        for j = 1:2^i
            r(:, j) = mean(d(:, find(ind == j)), 2); %#ok<FNDSB>
            x = disteu(d(:, find(ind == j)), r(:, j)); %#ok<FNDSB>
            for q = 1:length(x)
                t = t + x(q);
            end
        end
        if (((dpr - t)/t) < e)
            break;
        else
            dpr = t;
        end
    end
end
end
%--------------------------------------------------------------------------

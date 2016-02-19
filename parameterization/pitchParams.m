function [allParams] = pitchParams(y, Fs)

okna = 1;                                               % dlugosc okna
Nparams = 4;                                            % liczba parametrow dla kazdego filtru

N = floor(length(y)/(okna*Fs));

Params = zeros(Nparams, N);

%% filters
NumHigh1200 = [0.00511912773686704 0.00789153833692803 -0.0131116949979732 -0.103784096216857 -0.236101557379746 0.703596866478154 -0.236101557379746 -0.103784096216857 -0.0131116949979732 0.00789153833692803 0.00511912773686704];
NumHigh1600 = [4.36410195307487e-18 0.0126981174729760 0.0248019135288458 -0.0637871498077047 -0.276018100203916 0.599745691319317 -0.276018100203916 -0.0637871498077047 0.0248019135288458 0.0126981174729760 4.36410195307487e-18];
NumLow840 = [-0.000921075975485055 0.00743935130410590 0.0447888697367247 0.121570567349411 0.205732485747952 0.242779603674582 0.205732485747952 0.121570567349411 0.0447888697367247 0.00743935130410590 -0.000921075975485055];

fts = [NumLow840; NumHigh1200; NumHigh1600];
allParams = [];
%%
for j = 1:size(fts,1)
    nums = fts(j,:);
    out_y = filter(nums, 1, y);
    
    for i = 1:N
        [Pitch, ~] = yaapt(out_y((i-1)*(okna*Fs)+1:i*(okna*Fs)), Fs, 1, [], 0);
        Pitch = Pitch(Pitch~=0);                              % voiced pitch
        Par1 = max(Pitch)/min(Pitch);                         % Par1
        Par2 = std(Pitch)/mean(Pitch);                        % Par2
        Par3 = skewness(Pitch);                               % skewness
        Par4 = kurtosis(Pitch);                               % kurtosis
        Params(:,i) = [Par1; Par2; Par3; Par4];
    end
    allParams = [allParams; Params];
    
end

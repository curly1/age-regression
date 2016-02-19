function [newMFCCs, newDatabase] = utterancePartition(database)

%% Generowanie 10 nowych macierzy MFCC dla nagran dluzszych niz 30 sekund. 
% Kazda nowa macierz stanowi x% ramek MFCC bazowej macierzy MFCC.
% Za kazdym razem x% ramek jest losowanych z bazowej macierzy MFCC.
% Dla kazdego nagrania > 30 s otrzymujemy N+1 macierzy MFCC.

%%
disp('Generating more data...')

MFCCs = database.MFCC_delta_cms;
proc    = 0.5;
N       = 10;
minSec  = 30;

refSecMin = ones(size(database,1),1)*minSec;        
refSecMax = proc*database.duration_sec;

k = refSecMin < refSecMax;                      % nagrania dluzsze niz 30 sekund

licz = 1;
for i = 1:size(database,1)
    if k(i)
        nFrames = size(MFCCs{i,1},2);
        for j = 1:N+1
            if j == 1
                newMFCCs{licz,1} = MFCCs{i,1}(:,:);
                newDatabase(licz,:) = database(i,{'file_id', 'duration_sec', 'gender', 'age', 'age_class'});
                licz = licz + 1;
            else
                w = randperm(nFrames, round(nFrames*proc));
                newMFCCs{licz,1} = MFCCs{i,1}(:,w);
                newDatabase(licz,:) = database(i,{'file_id', 'duration_sec', 'gender', 'age', 'age_class'});
                newDatabase.duration_sec(licz) = database.duration_sec(i)*proc;
                licz = licz + 1;
            end
        end
    else
        newMFCCs{licz,1} = MFCCs{i,1}(:,:);
        newDatabase(licz,:) = database(i,{'file_id', 'duration_sec', 'gender', 'age', 'age_class'});
        licz = licz + 1;
    end
end

disp('More data generated.')
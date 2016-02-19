function [newMFCCs, newDatabase] = utterancePartition2(database)

%% Dzieli nagrania na ok. 30 sekundowe okna. Tylko dla nagran powyej 60 sekund.
% Np. dla 60-sekundowego nagrania otrzymujemy 3 macierze MFCC: oryginalna,
% pierwsze 30 s nagrania, drugie 30 sekund nagrania.

%%
disp('Generating more data...')

MFCCs = database.MFCC_delta_cms;
minSec  = 30;

k = database.duration_sec >= minSec*2;

licz = 1;
for i = 1:size(database,1)
    if k(i)
        nFrames = size(MFCCs{i,1},2);
        nUtt = floor(database.duration_sec(i)/minSec);
        for j = 1:nUtt+1
            if j == 1
                newMFCCs{licz,1} = MFCCs{i,1}(:,:);
                newDatabase(licz,:) = database(i,{'file_id', 'duration_sec', 'gender', 'age', 'age_class'});
                licz = licz + 1;
            else
                w = floor(nFrames/nUtt);
                try newMFCCs{licz,1} = MFCCs{i,1}(:,(j-2)*w+1:(j-1)*w);
                catch warning('w')
                end
                newDatabase(licz,:) = database(i,{'file_id', 'duration_sec', 'gender', 'age', 'age_class'});
                newDatabase.duration_sec(licz) = w/100;
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
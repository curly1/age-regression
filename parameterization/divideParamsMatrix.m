function newDatabase = divideParamsMatrix( database )

%% podzial macierzy MFCC na wektory cech MFCC. Kazdy wektor bedzie sluzyl do regresji
%%

n = 1;
newDatabase = table;
for i = 1:size(database,1)
  for j = 1: size(database.PitchParams{i},2)
    newDatabase.file_id(n,1) = database.file_id(i,1);
    newDatabase.duration_sec(n,1) = database.duration_sec(i,1);
    newDatabase.PitchParams{n,1} = database.PitchParams{i,1}(:,j);
    newDatabase.labels(n,1) = database.labels(i,1);
    n = n+1;
  end

end
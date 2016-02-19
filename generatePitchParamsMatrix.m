function [features, labels] = generatePitchParamsMatrix(database)

%% przeksztalcenie danych do postaci fetures/labels 

features = zeros(size(database,1), size(database.PitchParams{1},1));
labels = zeros(size(database,1), 1);

for i = 1:size(database,1)
    features(i,:) = database.PitchParams{i}';
    labels(i,1) = database.labels(i,1);
end

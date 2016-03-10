m=1:770;
allids = '';
for i = 1:length(m)
    i
    ids = cell(size(database.MFCC_delta_cms{i,1},2),1);
    for n = 1:size(database.MFCC_delta_cms{i,1},2);
        ids{n,1} = database.file_id{i};
    end
    allids = [allids; ids];
end

%%
m=1:770;
allidx = [];

for i = 1:length(m)
    i
    idx = zeros(size(database.MFCC_delta_cms{i,1},2),1);
    for n = 1:size(database.MFCC_delta_cms{i,1},2);
        idx(n,1) = m(i);
    end
    allidx = [allidx; idx];
end

%%
m=1:770;
all = [];

for i = 1:length(m)
    i
    a = zeros(size(database.MFCC_delta_cms{i,1},2),450);
    for n = 1:size(database.MFCC_delta_cms{i,1},2);
        a(n,:) = database.MFCC_delta_cms{i,1}(:,n);
    end
    all = [all; a];
end

%%
m=1:770;
alllabels = [];
for i = 1:length(m)
    i
    labels = zeros(size(database.MFCC_delta_cms{i,1},2),1);
    for n = 1:size(database.MFCC_delta_cms{i,1},2);
        labels(n,1) = database.age(i);
    end
    alllabels = [alllabels; labels];
end

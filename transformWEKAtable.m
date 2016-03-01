function [d1, d2] = transformWEKAtable(WEKAdatabase)

WEKAdatabase = sortrows(WEKAdatabase,'name','ascend');
n = height(WEKAdatabase);

all_file_id = cell(n,1);

for i = 1:n
    all_file_id{i,1} = [WEKAdatabase.name{i,1}(1:4) '.wav'];
end

[file_id,idx] = unique(all_file_id);
age_class = WEKAdatabase.agegroup(idx);

s = length(idx);                % liczba mowcow
database = table(file_id);
k = 1;


for i = 1:s
    i
    if i ~= s
        u = idx(i+1)-idx(i);    % liczba nagran dla kazdego mowcy
    else
        u = n-idx(s);
    end
    params = zeros(450,u);
    for j = 1:u
       params(:,j) = WEKAdatabase{k,2:451}';
       k = k+1;
    end
    MFCC_delta_cms{i,1} = params;
end

a = table(age_class);
d1 = [database, MFCC_delta_cms, a];
d1.Properties.VariableNames{2} = 'MFCC_delta_cms';

MFCCcms = cell(size(MFCC_delta_cms));
for i = 1:height(d1)
    MFCCcms{i,1} = cms(d1.MFCC_delta_cms{i});
end

d2 = [database, MFCCcms, a];
d2.Properties.VariableNames{2} = 'MFCC_delta_cms';

end
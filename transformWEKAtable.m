function [d1, d2] = transformWEKAtable(WEKAdatabase)

WEKAdatabase = sortrows(WEKAdatabase,'name','ascend');
n = height(WEKAdatabase);

all_file_id = cell(n,1);

for i = 1:n
    all_file_id{i,1} = [WEKAdatabase.name{i,1}(1:4) '.wav'];
end

[file_id,idx] = unique(all_file_id);
age_class = WEKAdatabase.agegroup(idx);
age = WEKAdatabase.age(idx);
gender = WEKAdatabase.gender(idx);

s = length(idx);                % liczba mowcow
database = table(file_id);
k = 1;


for i = 1:s
    i
    if i ~= s
        u = idx(i+1)-idx(i);    % liczba nagran dla kazdego mowcy
    else
        u = n+1-idx(s);
    end
    params = zeros(450,u);
    for j = 1:u
       params(:,j) = WEKAdatabase{k,2:451}';
       k = k+1;
    end
    WEKAPar{i,1} = params;
end

a = table(age_class);
b = table(age);
c = table(gender);
d1 = [database, WEKAPar, a, b, c];
d1.Properties.VariableNames{2} = 'MFCC_delta_cms';

WEKAPar_cms = cell(size(WEKAPar));
normWEKAPar_cms{i,1} = cell(size(WEKAPar));
for i = 1:height(d1)
    WEKAPar_cms{i,1} = cms(d1.MFCC_delta_cms{i});
    normWEKAPar_cms{i,1} = normPar(WEKAPar_cms{i,1});
end

d2 = [database, normWEKAPar_cms, a, b, c];
d2.Properties.VariableNames{2} = 'MFCC_delta_cms';

end
function databaseWithoutNaN = checkDatabaseNaN(database)

N = length(database.MFCC_delta_cms);
a = zeros(N,1);
b = zeros(N,1);

for i = 1:N
  a(i,1)= sum(sum(sum(isnan(database.MFCC_delta_cms{i,1}))));
  b(i,1)= isempty(database.MFCC_delta_cms{i,1});
end

idx = (a==0) & (b==0);

databaseWithoutNaN = database(idx,:);

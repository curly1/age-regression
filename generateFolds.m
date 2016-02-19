function [folds,who] = generateFolds(f, database)

[~,b] = unique(database.file_id);
[~,a] = unique(b); 

for i = 1:length(f)
  try folds(b(i):b(i+1)-1) = f(i);
      who(b(i):b(i+1)-1) = a(i);
  catch folds(b(i):height(database)) = f(i);
      who(b(i):height(database)) = a(i);
  end
end

folds = folds';
who = who';
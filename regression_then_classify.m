addpath([cd '/code'])

Y_true = Y_true;
Y_pred = Y_pred;

trueClass = zeros(size(Y_true,1),1);
predClass = zeros(size(Y_pred,1),1);

for i = 1:size(trueClass,1)
  trueClass(i,1) = determineAgeClass(Y_true(i));
end

for i = 1:size(predClass,1)
  predClass(i,1) = determineAgeClass(Y_pred(i));
end

ACC = sum(trueClass == predClass)/length(trueClass);

rmpath([cd '/code'])

%load('arff450.mat')

clear a
clear b
clear c
clear d
clear mowcy

mowcy = data.allidx(test_idx);
ajdi = data.allids(test_idx);
mowcy = mowcy(idx);
ajdi = ajdi(idx);
[a,b] = unique(mowcy);
[c,e] = sort(b);

for i = 1:length(c)
    
    figure()
    ajdi(c(i))
    Y_true1 = Y_true(c(i):c(i+1));
    Y_pred1 = Y_pred(c(i):c(i+1));
    plot(Y_true1, '-o'); hold on
    plot(Y_pred1, 'or')
    legend('True', 'Predicted')
    ylabel('age'), xlabel('# speaker')
    MAE = 1/(length(Y_pred1))*sum(abs(Y_pred1-Y_true1))
    clear Y_true1
    clear Y_pred1
    waitforbuttonpress;
    close all
    
end
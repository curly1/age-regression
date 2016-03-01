i = 1;
j = 1;

while true
    s = size(database.MFCC_delta_cms{j},2);
    if s > 1
    for n=1:s
        all(i,:) = database.MFCC_delta_cms{j}(:,n);
        i=i+1;
        if n == s
            j = j+1;
        end
    end
    else
        all(i,:) = database.MFCC_delta_cms{j};
        i=i+1;
        j=j+1;
    end
end
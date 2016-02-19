%% compares matrices and returns indexes of common part for matrix1
% matrix1 and matrix2 have to be sorted!

function [id1, id2] = compareMatrices(matrix1, matrix2)

N1 = length(matrix1);
N2 = length(matrix2);

k = min(N1,N2);
if k == N1
    m1 = matrix2;
    m2 = matrix1;
else if k == N2
        m1 = matrix1;
        m2 = matrix2;
    end
end

N11 = length(m1);
N21 = length(m2);

idx1 = logical((zeros(N11,1)));
idx2 = logical((zeros(N21,1)));

for i = 1:N11
  for j = 1:N21
    if m1(i) == m2(j)
      idx1(i) = true;
      idx2(j) = true;
      break
    end
  end
end

if k == N1, id1 = idx1; id2 = idx2; 
else if k == N2, id1 = idx2; id2 = idx1;
    end
end
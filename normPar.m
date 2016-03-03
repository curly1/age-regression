function normMatrixPar = normPar(MatrixPar)

minimum = 0;
maximum = 1;

normMatrixPar = zeros(size(MatrixPar));

for i = 1:size(MatrixPar,1)
    y = MatrixPar(i,:);
    y_norm = (y-min(y))*(maximum-minimum)/(max(y)-min(y))+minimum;
    normMatrixPar(i,:) = y_norm;
end

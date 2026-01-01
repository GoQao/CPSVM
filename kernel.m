
function K = kernel(X,Y,type,gamma,p)
[nb_data,~] = size(X);
switch type
    case 'linear'
        K = X*Y';
    case 'rbf'
%         gamma = gamma*gamma;
        XX = sum(X.*X,2)*ones(1,size(Y,1));
        YY = sum(Y.*Y,2)*ones(1,nb_data);
        K = XX+ YY' - 2*X*Y';
        K = exp(-K./(2*gamma));
    case 'poly'
        K = X*Y';
        K = (K + 1).^p;
end
end
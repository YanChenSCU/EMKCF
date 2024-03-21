function [U, Vs, alpha, objHistory] = EMKCF_neg_U_conerges(Ks, nCluster)

nKernel = length(Ks);
nSmp = size(Ks{1}, 1);


%*******************************************
% Init alpha
%*******************************************
aa = zeros(nKernel, 1);
tmp = Ks{1};
aa(1) = sum(diag(tmp));
for iKernel = 2:nKernel
    aa(iKernel) = sum(diag(Ks{iKernel}));
    tmp = tmp + Ks{iKernel};
end
tmp = (tmp + tmp')/2;
if nSmp < 200
    opt.disp = 0;
    [U, ~] = eigs(tmp, nCluster, 'la', opt);
else
    tmp = tmp - spdiags(diag(tmp), 0, nSmp, nSmp);
    U = spectral_cluster_pm_k(tmp, nCluster);
end
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, nCluster);
label = litekmeans(U_normalized, nCluster, 'MaxIter', 100, 'Replicates', 10);
U = ind2vec(label')' + 0.2;
clear tmp;

es = ones(1, nKernel);
alpha = sqrt(es)/sum(sqrt(es));

converges = false;
iter = 0;
maxIter = 100;
objHistory = [];
while ~converges
    iter = iter + 1;
    
    %*******************************************
    % Optimize Vs
    %*******************************************
    Vs = cell(1, nKernel);
    KVs = cell(1, nKernel);
    for iKernel = 1:nKernel
        KYi = Ks{iKernel}' * U; % m * c
        [U2, ~, V2] = svd(KYi, 'econ');
        Vs{iKernel} = U2 * V2'; % m * c
        KVs{iKernel} = Ks{iKernel} * Vs{iKernel}; % n * c
    end
    %     obj = compute_obj(Ks, Y, Vs, alpha);
    %     objHistory = [objHistory; obj]; %#ok
    
    
    %*******************************************
    % Optimize Y
    %*******************************************
    B = zeros(nSmp, nCluster);
    for iKernel = 1:nKernel
        B = B + (1/alpha(iKernel)) * KVs{iKernel};
    end
    B = -B;
    A = 1/alpha(1) * Ks{1};
    for iKernel = 2:nKernel
        A = A + (1/alpha(iKernel)) * Ks{iKernel};
    end
    A_pos = max(A, 0);
    A_neg = (abs(A) - A)/2;
    for iter2 = 1:200
        A_posU = A_pos * U;
        A_negU = A_neg * U;
        tmp1 = (sqrt(B.^2 + 4 * (A_posU .* A_negU)) - B);
        U = U .* tmp1 ./ max(2 * A_posU, eps);
    end
    %     obj = compute_obj(Ks, Y, Vs, alpha);
    %     objHistory = [objHistory; obj]; %#ok
    
    
    %*******************************************
    % Optimize alpha
    %*******************************************
    ab = zeros(nKernel, 1);
    bb = zeros(nKernel, 1);
    for iKernel = 1:nKernel
        ab(iKernel) = sum(sum( U .* KVs{iKernel}));
        KU = Ks{iKernel} * U;
        bb(iKernel) = sum(sum( U .* KU));
    end
    es = aa - 2*ab + bb;
    alpha = sqrt(es)/sum(sqrt(es));
    %*******************************************
    % Compute obj
    %*******************************************
    obj = sum(sum((1./alpha) .* es));
    % obj2 = compute_obj(Ks, Y, Vs, alpha);
    objHistory = [objHistory; obj]; %#ok
    
    
%     if (iter>20) && (  abs( objHistory(iter-1) - objHistory(iter) )  /abs(objHistory(iter-1) ) <1e-8 || iter>maxIter)
%         converges = true;
%     end
%     
    
    if iter > maxIter
        converges = true;
    end
end

end


function obj = compute_obj(Ks, Y, Vs, alpha)
obj = 0;
for iKernel = 1:length(Ks)
    KV = Ks{iKernel} * Vs{iKernel}; % n * c
    o = sum(diag(Ks{iKernel})) - sum(sum( Y .* KV));
    obj = obj + alpha(iKernel) * o;
end
end


function a = lp_simplex_proj(h, alpha)
% This function solve the following problem
%
%   \min_{a} \quad \sum_i a_i h_i = a^T h
%    s.t.    a_i >=0, \sum_i a_i^alpha = 1
%
% [1]Weighted Feature Subset Non-negative Matrix Factorization and Its Applications to Document Understanding. ICDM 2010
%

assert( (alpha > 0) && (alpha < 1), 'alpha should be (0, 1)');
t1 = 1 / alpha;
t2 = 1 / (alpha - 1);
t3 = alpha / (alpha - 1);

t4 = sum(h .^ t3);
t5 = (1 / t4)^t1;
t6 = h .^ t2;
a = t5 * t6;
end
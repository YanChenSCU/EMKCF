function paramCell = EEMKCF_build_param(ks)

if ~exist('ks', 'var')
    ks = [5, 10, 15, 20];
end

nParam = length(ks);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(ks)
    param = [];
    param.k = ks(i1);
    idx = idx + 1;
    paramCell{idx,1} = param;    
end
end
